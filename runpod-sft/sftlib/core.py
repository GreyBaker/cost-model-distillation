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
CONFIGS = ROOT / "configs"
SPEC = ROOT / "assets_spec.json"
STEPS = {"student": 100000, "teacher": 8000}
PINS = {"student": ("google/t5-small-lm-adapt", "4ff52753a4ddd906de166c573db99280719b22f9"),
        "teacher": ("google/t5-xl-lm-adapt", "f9843a55f0bdc37c957ddba20be0be4870b0a994"),
        "xsum": ("EdinburghNLP/xsum", "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9")}
KEYS = {"role", "run_name", "max_steps", "seed", "data_seed", "prefix", "max_source_tokens",
        "max_target_tokens", "microbatch_size", "gradient_accumulation_steps", "learning_rate",
        "warmup_steps", "dropout", "gradient_checkpointing", "save_every_steps", "keep_checkpoints",
        "eval_every_steps", "eval_examples", "evaluation_batch_size", "gpu_memory_warning_gib"}
TEACHER_KEYS = {"recipe", "scale_parameter", "cooldown_start_step", "loss_normalizing_factor",
                "z_loss"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(directory, excluded=()):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not owned portable artifact files")
        name = path.relative_to(directory).as_posix()
        if path.is_file() and name not in excluded:
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return files


def verify_tree(directory, files):
    directory = Path(directory).resolve()
    if not files:
        raise ValueError("Empty artifact manifest")
    for name, entry in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.is_symlink():
            raise ValueError("Unsafe artifact path")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Artifact integrity failure: {path}")


def sync_tree(directory):
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
        raise OSError(f"Need {required / 1024**3:.1f} GiB free; have {free / 1024**3:.1f} GiB")


def configure_local_storage(work):
    """Operational commands only: keep library caches/temp on the job's disk.

    With POD_DISK, use one shared cache tree there and require work underneath it.
    Without it, use the explicit work directory; never default temp/cache to an
    assumed second filesystem. Read-only commands must not call this function.
    """
    import tempfile
    work = Path(work).expanduser().resolve()
    if os.environ.get("POD_DISK"):
        disk = Path(os.environ["POD_DISK"]).expanduser().resolve(strict=True)
        if not disk.is_dir() or not work.is_relative_to(disk):
            raise ValueError("Work directory must be inside the existing POD_DISK directory")
    else:
        disk = work
    base = disk / ".runtime"
    cache = base / "cache"
    values = {
        "XDG_CACHE_HOME": cache,
        "PIP_CACHE_DIR": cache / "pip",
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "TORCH_HOME": cache / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TMPDIR": base / "tmp", "TMP": base / "tmp", "TEMP": base / "tmp",
    }
    for name, path in values.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(base / "tmp")
    return {name: str(path) for name, path in values.items()}


def config(role, directory=CONFIGS):
    if role not in STEPS:
        raise ValueError("Unknown preparation role")
    value = read_json(Path(directory) / f"{role}.json")
    expected = KEYS | (TEACHER_KEYS if role == "teacher" else set())
    if set(value) != expected:
        raise ValueError(f"Missing/unknown config fields: {set(value) ^ expected}")
    if value["role"] != role or value["max_steps"] != STEPS[role]:
        raise ValueError("Role/step budget must match Appendix A.2 (student 100K, teacher 8K)")
    name = value["run_name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Unsafe run name")
    for key in ("max_steps", "max_source_tokens", "max_target_tokens", "microbatch_size",
                "gradient_accumulation_steps", "save_every_steps", "keep_checkpoints",
                "eval_every_steps", "eval_examples", "evaluation_batch_size"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "data_seed", "warmup_steps"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("learning_rate", "gpu_memory_warning_gib", "dropout"):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not math.isfinite(value[key]):
            raise ValueError(f"Invalid {key}")
    if value["learning_rate"] <= 0 or value["gpu_memory_warning_gib"] <= 0 or not 0 <= value["dropout"] < 1:
        raise ValueError("Invalid LR/VRAM advisory/dropout")
    if value["keep_checkpoints"] != 2 or value["warmup_steps"] > value["max_steps"]:
        raise ValueError("Retain two recent checkpoints (preserving historical pins); warmup must fit budget")
    if type(value["gradient_checkpointing"]) is not bool or not isinstance(value["prefix"], str):
        raise ValueError("Invalid gradient checkpointing/prefix")
    if role == "teacher":
        if value["recipe"] != "t5x_xsum_xl_adapted_v1" or value["scale_parameter"] is not True:
            raise ValueError("Teacher requires the declared parameter-scaled researched recipe")
        for key in ("cooldown_start_step", "loss_normalizing_factor"):
            if type(value[key]) is not int or value[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not value["warmup_steps"] <= value["cooldown_start_step"] < value["max_steps"]:
            raise ValueError("Invalid teacher cooldown boundaries")
        if (type(value["z_loss"]) not in (int, float) or not math.isfinite(value["z_loss"])
                or value["z_loss"] < 0):
            raise ValueError("Invalid teacher z-loss coefficient")
    return value


def asset_spec(value):
    spec = read_json(SPEC)
    for name, pin in PINS.items():
        if (spec[name]["repo_id"], spec[name]["revision"]) != pin:
            raise ValueError("Base/data revision pins changed")
        for item in spec[name]["files"]:
            if Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
                raise ValueError("Unsafe pinned asset path")
    return {"model": spec[value["role"]], "xsum": spec["xsum"]}


def check_teacher_work(work, value):
    """Refuse mixing old/new teacher experiments, including standalone runs."""
    if value["role"] != "teacher":
        return
    from .resume import config_compatible
    for path in sorted((Path(work) / "runs").glob("*/run.json")):
        if not config_compatible(read_json(path)["config"], value):
            raise ValueError("Existing teacher uses a different recipe/run. Preserve it and use a NEW SFT_WORK "
                             "containing only the restored student; do not resume old teacher weights.")


def recipe_evidence(value):
    if value["role"] != "teacher":
        return None
    return {
        "recipe": value["recipe"],
        "source": "https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/"
                  "7ea890cd306ab924572bb2aaadadbfbc410032b3/configs/finetune_t511_xl_xsum.gin",
        "public_precedent": "batch256, 9K updates, scaled Adafactor coefficient5e-4, "
                            "dropout0.1, train512/64, inference1024/64, normalizer233472",
        "adaptations": f"batch{value['microbatch_size'] * value['gradient_accumulation_steps']}, "
                       f"{value['max_steps']} updates, source{value['max_source_tokens']}, "
                       f"warmup{value['warmup_steps']}/cooldown{value['cooldown_start_step']}-{value['max_steps']}, "
                       "pinned LM-adapted XL with fresh optimizer, local fixed-shape unpacked data",
        "normalization": f"Summed valid-token CE plus z-loss{value['z_loss']} divided by"
                         f"{value['loss_normalizing_factor']} once per full update; "
                         "no batch-size, token-count or accumulation rescaling",
        "boundary": "Research-supported adaptation, not exact GKD-author SFT or a demonstrated optimum",
    }


def checkpoint_due(value, step, stopping=False):
    return stopping or step == value["max_steps"] or (step > 0 and step % value["save_every_steps"] == 0)


def lr_for_update(value, completed):
    warmup = min(1.0, (completed + 1) / max(1, value["warmup_steps"]))
    cooldown = 1.0
    if "cooldown_start_step" in value:
        # completed counts successful updates; the last update has a nonzero
        # coefficient, with the linear schedule reaching zero at the endpoint.
        cooldown = max(0.0, min(1.0, (value["max_steps"] - completed)
                              / (value["max_steps"] - value["cooldown_start_step"])))
    return value["learning_rate"] * warmup * cooldown


def data_signature(value):
    return {key: value[key] for key in ("prefix", "max_source_tokens", "max_target_tokens")}


def environment():
    return {"python": platform.python_version(), "platform": platform.system(),
            "packages": {name: metadata.version(name) for name in
                         ("torch", "transformers", "datasets", "accelerate", "numpy", "safetensors",
                          "sentencepiece", "rouge-score", "tokenizers", "huggingface-hub", "pyarrow")}}


def code_signature():
    paths = [ROOT / "prepare_models.py", *sorted((ROOT / "sftlib").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


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
    """Kernel-released lock, not a stealable PID lock."""
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
            raise RuntimeError(f"Another writer owns {self.path}") from error
        return self

    def __exit__(self, *args):
        self.stream.close()
