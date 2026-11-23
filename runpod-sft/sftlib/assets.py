from __future__ import annotations

import os
from pathlib import Path
import shutil
import urllib.request
import uuid

from .core import SPEC, asset_spec, atomic_json, check_space, read_json, sha256


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]


def transfer(url, destination, entry, local_source=None):
    if valid(destination, entry):
        return
    if destination.exists():
        raise ValueError(f"Refusing to replace corrupt existing asset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if local_source is not None:
        if partial.exists():
            raise FileExistsError(f"Inspect partial before importing: {partial}")
        shutil.copyfile(local_source, partial)
    else:
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > entry["bytes"]:
                    raise ValueError("Oversized partial download")
                if offset < entry["bytes"]:
                    headers = {"User-Agent": "t5-xsum-preparation/1.0"}
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
                            while chunk := response.read(4 * 1024**2):
                                stream.write(chunk)
                break
            except OSError:
                if attempt == 2:
                    raise
    if not valid(partial, entry):
        raise ValueError(f"Asset hash mismatch; partial retained: {partial}")
    os.replace(partial, destination)


def acquire(work, value, from_assets=None):
    spec = asset_spec(value)
    base = work / "assets"
    base.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for name, group in spec.items():
        for entry in group["files"]:
            path = base / name / entry["path"]
            partial = path.with_name(path.name + ".part")
            current = path if path.exists() else partial
            remaining += max(0, entry["bytes"] - (current.stat().st_size if current.exists() else 0))
    check_space(base, remaining + 1024**3)
    for name, group in spec.items():
        for entry in group["files"]:
            source = None
            if from_assets:
                source = Path(from_assets) / (value["role"] if name == "model" else "xsum") / entry["path"]
            prefix = "datasets/" if name == "xsum" else ""
            url = f"https://huggingface.co/{prefix}{group['repo_id']}/resolve/{group['revision']}/{entry['path']}"
            transfer(url, base / name / entry["path"], entry, source)
    atomic_json(base / "manifest.json", {"status": "complete", "spec_sha256": sha256(SPEC), "assets": spec})


def verify(work, value):
    base = work / "assets"
    spec = asset_spec(value)
    manifest = read_json(base / "manifest.json")
    if manifest != {"status": "complete", "spec_sha256": sha256(SPEC), "assets": spec}:
        raise ValueError("Asset manifest/spec changed or incomplete")
    for name, group in spec.items():
        for entry in group["files"]:
            if not valid(base / name / entry["path"], entry):
                raise ValueError(f"Asset integrity failure: {name}/{entry['path']}")
    return manifest
