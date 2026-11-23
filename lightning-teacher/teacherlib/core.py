from __future__ import annotations

import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "teacher_8k.json"
SPEC = ROOT / "assets_spec.json"
CONFIG_KEYS = {"run_name", "max_steps", "seed", "data_seed", "prefix", "max_source_tokens",
               "max_target_tokens", "microbatch_size", "gradient_accumulation_steps", "learning_rate",
               "warmup_steps", "dropout", "gradient_checkpointing", "save_every_steps",
               "keep_checkpoints", "eval_every_steps", "eval_examples", "gpu_memory_warning_gib"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 ** 2):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(directory, excluded=()):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not portable/owned artifact files: {path}")
        if path.is_file() and path.relative_to(directory).as_posix() not in excluded:
            result[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def verify_tree(directory, files):
    directory = Path(directory).resolve()
    if not files:
        raise ValueError("Empty artifact manifest")
    for name, entry in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError("Unsafe artifact path")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Artifact hash/size failure: {path}")


def sync_tree(directory):
    """Request filesystem flush before publishing a checkpoint completion marker."""
    for path in Path(directory).rglob("*"):
        if path.is_file():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())


def sync_directory(directory):
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def check_space(directory, required):
    path = Path(directory)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < required:
        raise OSError(f"Need {required / 1024**3:.1f} GiB free, have {free / 1024**3:.1f} GiB: {directory}")


def config(path=CONFIG):
    value = read_json(path)
    expected = CONFIG_KEYS
    if set(value) != expected:
        raise ValueError(f"Missing/unknown config keys: {set(value) ^ expected}")
    if not value["run_name"] or Path(value["run_name"]).name != value["run_name"] or value["run_name"] in (".", ".."):
        raise ValueError("run_name must be one safe directory name")
    for key in ("max_steps", "microbatch_size", "gradient_accumulation_steps", "max_source_tokens",
                "max_target_tokens", "save_every_steps", "keep_checkpoints", "eval_every_steps", "eval_examples"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "data_seed", "warmup_steps"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if value["keep_checkpoints"] != 2:
        raise ValueError("This bounded storage protocol retains exactly two complete checkpoints")
    for key in ("learning_rate", "gpu_memory_warning_gib"):
        if isinstance(value[key], bool) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= value["dropout"] < 1 or not 0 <= value["warmup_steps"] <= value["max_steps"]:
        raise ValueError("Invalid dropout/warmup")
    if type(value["gradient_checkpointing"]) is not bool or not isinstance(value["prefix"], str):
        raise ValueError("Invalid checkpointing/prefix")
    return value


def checkpoint_due(value, step, stopping=False):
    """Only optimizer-step progress/pause/final completion triggers a save."""
    return stopping or step == value["max_steps"] or (step > 0 and step % value["save_every_steps"] == 0)


def lr_for_update(value, completed_steps):
    return value["learning_rate"] * min(1.0, (completed_steps + 1) / max(1, value["warmup_steps"]))


def data_signature(value):
    return {key: value[key] for key in ("prefix", "max_source_tokens", "max_target_tokens")}


def environment():
    return {"python": platform.python_version(), "platform": platform.system(),
            "packages": {name: metadata.version(name) for name in
                         ("torch", "transformers", "datasets", "accelerate", "numpy", "safetensors",
                          "sentencepiece", "rouge-score", "tokenizers", "huggingface-hub", "pyarrow")}}


def code_signature():
    paths = [ROOT / "teacher.py", *sorted((ROOT / "teacherlib").glob("*.py"))]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


class Cursor:
    def __init__(self, size, seed, position=0):
        if size < 1 or position < 0:
            raise ValueError("Invalid data cursor")
        self.size, self.seed, self.position = size, seed, position
        self.epoch, self.order = None, []

    def take(self, count):
        result = []
        for _ in range(count):
            epoch, offset = divmod(self.position, self.size)
            if epoch != self.epoch:
                self.order = list(range(self.size))
                random.Random(self.seed + epoch).shuffle(self.order)
                self.epoch = epoch
            result.append(self.order[offset])
            self.position += 1
        return result


class RunLock:
    """OS lock releases on process death; the tiny lock file can remain safely."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError(f"Another process owns the work directory lock: {self.path}") from error
        return self

    def __exit__(self, *exc):
        self.stream.close()
