from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "runpod-sft"))
sys.path.insert(0, str(REPO / "runpod-gkd"))

from sftlib.core import (RunLock, atomic_json, check_space, environment as base_environment, read_json, sha256,
                         sync_directory, sync_tree, tree_manifest, verify_tree)

PROFILE = "native-t5-5.20.2-v1"
PAIR_KIND = "validated-external-pair-v1"
ARTIFACT_KIND = "external-artifact-v1"
OOM_EXIT = 86


def environment():
    from importlib.metadata import distributions
    info = base_environment()
    info["installed_distributions"] = dict(sorted(
        (dist.metadata["Name"].lower().replace("_", "-"), dist.version) for dist in distributions()
        if dist.metadata["Name"]))
    return info


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code():
    paths = [ROOT / "external_teacher.py", *sorted((ROOT / "external_workflow").glob("*.py"))]
    paths += [*sorted((REPO / "runpod-sft/sftlib").glob("*.py")),
              *sorted((REPO / "runpod-gkd/gridlib").glob("*.py"))]
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}


def owned(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if (Path(relative).is_absolute() or "\\" in str(relative) or ".." in Path(relative).parts
            or not path.resolve().is_relative_to(root)):
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Symlinks are not owned artifacts")
    return path


def operational_paths(disk, work, grid, restored=None, reference=None):
    disk = Path(disk).expanduser().resolve(strict=True)
    if not disk.is_dir():
        raise ValueError("POD_DISK must be an existing directory")
    paths = [Path(work).resolve(), Path(grid).resolve(), ROOT.resolve(),
             (REPO / "runpod-sft").resolve(), (REPO / "runpod-gkd").resolve()]
    paths += [Path(restored).resolve()] if restored is not None else []
    for path in paths:
        if not path.is_relative_to(disk):
            raise ValueError("Code, environments, work and restored inputs must be on POD_DISK")
        if any(p.name.lower() == "results_and_temp" for p in (path, *path.parents)):
            raise ValueError("results_and_temp is immutable; transfer a copy to the Pod first")
    work, grid = paths[:2]
    if reference is not None:
        # A venv Python is normally a symlink to the system interpreter. Check
        # its owned parent, but preserve the venv invocation path.
        executable = Path(reference).expanduser().absolute()
        if not executable.parent.resolve().is_relative_to(disk):
            raise ValueError("Reference virtualenv must be on POD_DISK")
    for a, b in ((work, grid), (work, REPO), (grid, REPO)):
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError("Use separate source, external work and grid directories")
    if restored is not None:
        restored = Path(restored).resolve()
        if any(restored.is_relative_to(p) or p.is_relative_to(restored) for p in (work, grid)):
            raise ValueError("Restored SFT root must be separate and read-only")
    for path in [*paths, *([executable.parent.resolve()] if reference is not None else [])]:
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.stat().st_dev != disk.stat().st_dev:
            raise ValueError("Nested mounts on another device violate the one-disk workflow")
    for root in (work, grid):
        if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("Work/grid directories must not contain symlinks to unowned files")
    for relative in (".runtime", ".runtime/cache", ".runtime/cache/pip", ".runtime/cache/huggingface",
                     ".runtime/cache/huggingface/hub", ".runtime/cache/huggingface/datasets",
                     ".runtime/cache/torch", ".runtime/cache/torchinductor", ".runtime/cache/triton",
                     ".runtime/cache/cuda", ".runtime/tmp"):
        path = disk / relative
        if path.is_symlink() or not path.resolve().is_relative_to(disk):
            raise ValueError("Runtime cache/temp roots must be owned on POD_DISK")
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.stat().st_dev != disk.stat().st_dev:
            raise ValueError("Runtime caches/temp must not use another mounted device")
    return disk, work, grid


def request(work):
    return read_json(Path(work) / "run.json")


def pair_root(work):
    path = Path(work) / "pair"
    if path.is_symlink() or not path.resolve().is_relative_to(Path(work).resolve()):
        raise ValueError("External input root must be owned, not a symlink")
    return path


def identity(work, stage, **extra):
    return {"definition_sha256": digest(request(work)["definition"]), "stage": stage, **extra}


def verify(path, expected=None, full=True):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Symlink artifact")
    marker = read_json(path / "COMPLETE.json")
    if marker.get("kind") != ARTIFACT_KIND or marker.get("status") != "complete":
        raise ValueError(f"Not a complete external artifact: {path}")
    if expected is not None and marker["identity"] != expected:
        raise ValueError(f"Artifact identity changed: {path}")
    actual = set()
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("Symlink inside a sealed artifact")
        if item.is_file() and item != path / "COMPLETE.json":
            actual.add(item.relative_to(path).as_posix())
    if actual != set(marker["files"]):
        raise ValueError("Sealed artifact contains missing or unmanifested files")
    if full:
        verify_tree(path, marker["files"])
    return marker


def temporary(final):
    final = Path(final)
    if any(p.is_symlink() for p in (final, *final.parents)):
        raise ValueError("Symlink in artifact destination")
    final.parent.mkdir(parents=True, exist_ok=True)
    path = final.parent / f".{final.name}-{uuid.uuid4().hex}.incomplete"
    path.mkdir()
    return path


def publish(temporary_path, final, expected, **metadata):
    """Publish once. Incomplete work is preserved, never mistaken for completion."""
    final, temporary_path = Path(final), Path(temporary_path)
    if final.exists():
        raise FileExistsError(f"Never overwrite an artifact: {final}")
    sync_tree(temporary_path)
    atomic_json(temporary_path / "COMPLETE.json", {
        "kind": ARTIFACT_KIND, "status": "complete", "identity": expected,
        "source_fingerprints": code(), "unix_time": time.time(),
        **metadata, "files": tree_manifest(temporary_path)})
    sync_directory(temporary_path)
    os.replace(temporary_path, final)
    sync_directory(final.parent)
    return verify(final, expected)


def cached(path, expected):
    if Path(path).exists():
        verify(path, expected)
        return True
    return False


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"unix_time": time.time(), **record}, allow_nan=False) + "\n")


def runtime(work, reference=False, gpu=False):
    """Version gates are semantic; source fingerprints never gate resume."""
    import importlib.metadata
    expected_version = "4.57.6" if reference else "5.20.2"
    if importlib.metadata.version("transformers") != expected_version:
        raise ValueError(f"Use the isolated Transformers {expected_version} environment")
    if os.environ.get("POD_DISK"):
        disk = Path(os.environ["POD_DISK"]).resolve(strict=True)
        prefix = Path(sys.prefix).resolve()
        if (sys.prefix == sys.base_prefix or not prefix.is_relative_to(disk)
                or prefix.stat().st_dev != disk.stat().st_dev):
            raise ValueError("Activate the isolated virtualenv located on POD_DISK")
    result = {"environment": environment()}
    device = None
    if gpu:
        import torch
        from sftlib.modeling import cuda_device
        device = cuda_device({"gpu_memory_warning_gib": 75}, for_training=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        info = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": info.name, "total_memory": info.total_memory,
                         "capability": list(torch.cuda.get_device_capability(device)),
                         "cuda": torch.version.cuda}
        path = Path(work) / ("reference_runtime.json" if reference else "native_runtime.json")
        if path.exists() and read_json(path) != result:
            raise ValueError("GPU/software changed since validation; use a new external work root")
        if not path.exists():
            atomic_json(path, result)
    return device, result
