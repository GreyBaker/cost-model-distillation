from __future__ import annotations

import os
from pathlib import Path
import shutil
import urllib.request
import uuid

from .core import SPEC, atomic_json, check_space, read_json, sha256


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]


def transfer(url, destination, entry, local_source=None):
    if valid(destination, entry):
        return
    if destination.exists():
        raise ValueError(f"Refusing to replace a corrupt existing asset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if local_source is not None:
        if partial.exists():
            raise FileExistsError(f"Inspect existing partial before importing: {partial}")
        shutil.copyfile(local_source, partial)
    else:
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > entry["bytes"]:
                    raise ValueError(f"Oversized partial: {partial}")
                if offset < entry["bytes"]:
                    headers = {"User-Agent": "t5-xl-xsum-sft/1.0"}
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    request = urllib.request.Request(url + "?download=true&request_id=" + uuid.uuid4().hex, headers=headers)
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if offset and response.status == 206:
                            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                                raise ValueError("Unexpected byte range")
                        elif response.status == 200:
                            offset = 0
                        else:
                            raise ValueError(f"Unexpected HTTP status: {response.status}")
                        print(f"Downloading {destination.name}: {offset:,}/{entry['bytes']:,}", flush=True)
                        with partial.open("ab" if offset else "wb") as stream:
                            while chunk := response.read(4 * 1024 ** 2):
                                stream.write(chunk)
                break
            except OSError:
                if attempt == 2:
                    raise
    if not valid(partial, entry):
        raise ValueError(f"Asset hash mismatch; partial retained: {partial}")
    os.replace(partial, destination)


def acquire(work, from_assets=None):
    spec = read_json(SPEC)
    base = work / "assets"
    base.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for name, group in spec.items():
        for entry in group["files"]:
            path = base / name / entry["path"]
            partial = path.with_name(path.name + ".part")
            current = path if path.exists() else partial
            remaining += max(0, entry["bytes"] - (current.stat().st_size if current.exists() else 0))
    check_space(base, remaining + 1024 ** 3)
    for name, group in spec.items():
        for entry in group["files"]:
            source = None
            if from_assets:
                source = from_assets / ("teacher" if name == "model" else "xsum") / entry["path"]
            prefix = "datasets/" if name == "xsum" else ""
            transfer(f"https://huggingface.co/{prefix}{group['repo_id']}/resolve/{group['revision']}/{entry['path']}",
                     base / name / entry["path"], entry, source)
    atomic_json(base / "manifest.json", {"status": "complete", "spec_sha256": sha256(SPEC), "assets": spec})


def verify(work, released_run_id=None):
    base = work / "assets"
    manifest = read_json(base / "manifest.json")
    spec = read_json(SPEC)
    if manifest != {"status": "complete", "spec_sha256": sha256(SPEC), "assets": spec}:
        raise ValueError("Asset manifest/spec changed or incomplete")
    for name, group in spec.items():
        for entry in group["files"]:
            path = base / name / entry["path"]
            if (name == "model" and entry["path"] == "pytorch_model.bin" and not path.exists()
                    and released_run_id is not None):
                receipt = read_json(work / "BASE_WEIGHT_RELEASED.json")
                if (receipt.get("run_id") != released_run_id or receipt.get("spec_sha256") != sha256(SPEC)
                        or receipt.get("weight") != entry):
                    raise ValueError("Invalid released-base receipt")
                continue
            if not valid(path, entry):
                raise ValueError(f"Asset integrity failure: {name}/{entry['path']}")
    return manifest


def release_base(work, run_dir, apply=False):
    """Explicitly free only our cached base weight after two verified restarts exist."""
    from . import checkpoints, data
    run = read_json(run_dir / "run.json")
    choices = checkpoints.completed(run_dir)
    if len(choices) < 2:
        raise ValueError("Keep the base until at least two complete restart checkpoints exist")
    for path in choices[-2:]:
        checkpoints.verify(path, run["run_id"])
    data.verify(work, run["config"])
    verify(work, run["run_id"])
    weight = next(x for x in read_json(SPEC)["model"]["files"] if x["path"] == "pytorch_model.bin")
    path = work / "assets" / "model" / weight["path"]
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("Refusing release through a symlinked asset path")
    result = {"operation": "release" if apply else "preview only", "path": str(path),
              "bytes_freed_if_present": weight["bytes"] if path.exists() else 0,
              "retained_restarts": [str(x) for x in choices[-2:]]}
    if apply and path.exists():
        atomic_json(work / "BASE_WEIGHT_RELEASED.json", {"run_id": run["run_id"],
                    "spec_sha256": sha256(SPEC), "weight": weight,
                    "checkpoint_markers": {x.name: sha256(x / "COMPLETE.json") for x in choices[-2:]}})
        path.unlink()  # One explicitly owned cached file, never another checkpoint/base directory.
    return result
