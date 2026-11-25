"""Pinned, resumable source downloads; imported only by an explicit Pod worker."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

from .common import (ROOT, atomic_json, cached, check_space, identity, owned, pair_root,
                     publish, read_json, request, sha256, temporary, verify)
from sftlib.console import progress


def matches(path, item):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"]:
        return False
    if "sha256" in item:
        return sha256(path) == item["sha256"]
    h = hashlib.sha1(f"blob {item['bytes']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == item["git_blob_sha1"]


def transfer(url, destination, item):
    destination = Path(destination)
    if destination.exists():
        if not matches(destination, item):
            raise ValueError(f"Existing source is corrupt; not overwritten: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink():
        raise ValueError("Unsafe download partial symlink")
    for attempt in range(5):
        start = partial.stat().st_size if partial.exists() else 0
        if start > item["bytes"]:
            raise ValueError("Download partial exceeds locked size")
        if start == item["bytes"]:
            break
        headers = {"Accept-Encoding": "identity", "User-Agent": "external-xsum-import/1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                status = response.status
                if start and status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Invalid resumed response range")
                elif status == 200:
                    start = 0  # Server ignored Range; restart only our unverified partial.
                else:
                    raise ValueError(f"Unexpected download HTTP status {status}")
                with partial.open("ab" if start else "wb") as stream, progress(
                        item["bytes"], initial=start, desc=destination.name, unit="B") as bar:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        bar.update(len(chunk))
                        if stream.tell() > item["bytes"]:
                            raise ValueError("Download exceeds locked size")
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size == item["bytes"]:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))
    if not matches(partial, item):
        raise ValueError(f"Downloaded source hash/size mismatch; partial preserved: {partial}")
    os.replace(partial, destination)


def source(work, label):
    return pair_root(work) / "sources" / label


def acquire(work):
    run = request(work)
    locked = read_json(Path(work) / "models.lock.json")
    names = ["teacher", "xsum"]
    if run["definition"]["student"] == "public":
        names += ["public_student"]
    for label in names:
        spec = locked["evaluation_data"] if label == "xsum" else locked["models"][label]
        expected = identity(work, "source", label=label, source=spec)
        final = source(work, label)
        if cached(final, expected):
            if any(not matches(owned(final, item["path"]), item) for item in spec["files"]):
                raise ValueError("Cached source no longer matches release hashes")
            continue
        # Deterministic, owned download area supports transfer resume across invocations.
        scratch = final.with_name(final.name + ".download")
        if scratch.is_symlink():
            raise ValueError("Unsafe download directory symlink")
        scratch.mkdir(parents=True, exist_ok=True)
        owner = scratch / "REQUEST.json"
        if owner.exists() and read_json(owner) != expected:
            raise ValueError("Partial download belongs to a different source")
        atomic_json(owner, expected)
        missing = sum(item["bytes"] for item in spec["files"] if not matches(owned(scratch, item["path"]), item))
        check_space(scratch, missing + 1024 ** 3)
        for item in spec["files"]:
            target = owned(scratch, item["path"])
            # Reuse only a verified copy of already-uploaded official raw data.
            restored = run["definition"].get("restored_root")
            local = (Path(restored) / "student/assets/xsum" / item["path"]) if restored and label == "xsum" else None
            if local is not None and matches(local, item) and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, target)
            else:
                url = f"https://huggingface.co/{'datasets/' if label == 'xsum' else ''}{spec['repo_id']}/resolve/{spec['revision']}/{item['path']}"
                transfer(url, target, item)
        atomic_json(scratch / "SOURCE.json", {"source": spec, "raw_sha256": {
            item["path"]: sha256(owned(scratch, item["path"])) for item in spec["files"]}})
        publish(scratch, final, expected, local_sft_updates=0)
    if run["definition"]["student"] == "restored":
        from sftlib.core import CONFIGS
        from sftlib.sequence import verified_student
        restored = Path(run["definition"]["restored_root"])
        result = verified_student(restored, CONFIGS)
        final = pair_root(work) / "reports/restored_source"
        expected = identity(work, "restored_source", checkpoint_complete_sha256=sha256(
            Path(result["fp32_checkpoint"]) / "COMPLETE.json"))
        if not cached(final, expected):
            scratch = temporary(final)
            atomic_json(scratch / "source.json", result)
            publish(scratch, final, expected)


def student_source(work):
    if request(work)["definition"]["student"] == "public":
        return source(work, "public_student")
    return Path(read_json(pair_root(work) / "reports/restored_source/source.json")["fp32_checkpoint"])


def verified_model_source(work, role):
    """Repeat raw-byte verification immediately before any deserialization."""
    run = request(work)
    if role == "student" and run["definition"]["student"] == "restored":
        from sftlib.core import CONFIGS
        from sftlib.sequence import verified_student
        result = verified_student(Path(run["definition"]["restored_root"]), CONFIGS)
        path = Path(result["fp32_checkpoint"])
        if sha256(path / "COMPLETE.json") != run["definition"]["restored_checkpoint_complete_sha256"]:
            raise ValueError("Restored student changed since experiment creation")
        return path
    label = "teacher" if role == "teacher" else "public_student"
    spec = read_json(Path(work) / "models.lock.json")["models"][label]
    path = source(work, label)
    verify(path, identity(work, "source", label=label, source=spec))
    if any(not matches(owned(path, entry["path"]), entry) for entry in spec["files"]):
        raise ValueError("Public source bytes differ from locked release")
    return path
