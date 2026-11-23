"""Plan or run independent, equal-runtime XSum GKD cells with local assets."""
from __future__ import annotations

import argparse
import contextlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from download_summarizer import MANIFEST, verify_download
from gkdproto.common import ROOT, atomic_json, load_rows, read_json, sha256

TEACHER = ROOT / "assets/jordiclive_flan_t5_3b_summarizer"
PREFIX = "Given the following news article, summarize the article in one sentence: "
PREPARED = ROOT / "prepared/xsum_gkd_full"
ORDER = [("off_policy", 0.0), ("on_policy", 1.0), ("mixed", 0.5)]


def recipe(name, student, microbatch, budget_seconds=None, algorithm="gkd", fraction=0.0, prepared=PREPARED):
    if type(microbatch) is not int or microbatch < 1 or 32 % microbatch:
        raise ValueError("Microbatch must divide effective batch 32")
    return {
        "run_name": name, "algorithm": algorithm, "student_path": str(student),
        "teacher_path": str(TEACHER), "prepared_dir": str(prepared),
        "prefix": PREFIX, "max_source_tokens": 1024, "max_target_tokens": 64,
        "microbatch_size": microbatch, "gradient_accumulation_steps": 32 // microbatch,
        "max_steps": 100000 if algorithm == "sft" else 40000,
        "schedule_horizon": 100000 if algorithm == "sft" else 40000,
        "warmup_steps": 2000, "learning_rate": 0.001, "optimizer": "adafactor",
        "beta": 0.0, "lmbda": fraction, "mixture_mode": "balanced",
        "off_policy_source": "reference", "sampling_temperature": 1.0,
        "precision": "bfloat16", "device": "cuda:0", "teacher_device": "cuda:0",
        "gradient_checkpointing": True, "pad_to_max_length": True,
        "compute_budget_seconds": budget_seconds, "budget_schedule": budget_seconds is not None,
        "cuda_memory_fraction": 0.88, "save_every": 50, "keep_checkpoints": 2,
        "record_code_signature": True, "seed": 17, "data_seed": 104746,
    }


def select_microbatch(results):
    """Pick fastest geometric-mean update time across both pure sources, not ROUGE."""
    choices = {}
    for size in sorted({item["microbatch"] for item in results}):
        matching = [item for item in results if item["microbatch"] == size]
        if {item["branch"] for item in matching if item["status"] == "complete"} == {"off_policy", "on_policy"}:
            times = [item["warmed_update_seconds"] for item in matching if item["status"] == "complete"]
            choices[size] = math.exp(statistics.mean(math.log(value) for value in times))
    if not choices:
        raise RuntimeError("No microbatch passed both off-policy and on-policy pilots; no full run will start")
    return min(choices, key=choices.get), choices


@contextlib.contextmanager
def keep_awake():
    """Prevent unattended Windows idle sleep for this job; restore on exit."""
    if os.name != "nt":
        yield
        return
    import ctypes
    function = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    if not function(0x80000000 | 0x00000001):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        function(0x80000000)


def prepare_data():
    if PREPARED.exists():
        manifest = read_json(PREPARED / "manifest.json")
        for split, count in (("train", 204045), ("validation", 11332)):
            item = manifest["subsets"][split]
            if item["count"] != count or sha256(PREPARED / f"{split}.jsonl") != item["sha256"]:
                raise ValueError("Existing full preparation is incompatible")
        if manifest["dataset"]["revision"] != "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9":
            raise ValueError("XSum revision changed")
        return
    raise FileNotFoundError("Prepare the full local XSum pool before training")


class Study:
    def __init__(self, directory, spec):
        self.directory, self.spec = directory, spec
        self.configs = directory / "configs"
        self.logs = directory / "logs"
        self.configs.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)

    def status(self, phase, **extra):
        value = {"phase": phase, "wall_time_unix": time.time(), **extra}
        atomic_json(self.directory / "status.json", value)
        print(json.dumps(value), flush=True)

    def command(self, name, arguments, allow_failure=False):
        log = self.logs / f"{name}-{time.time_ns()}.log"
        self.status(name, log=str(log))
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
        started = time.time()
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-u", *map(str, arguments)], cwd=ROOT,
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {"phase": name, "returncode": result.returncode, "log": str(log),
                  "wall_seconds": time.time() - started, "wall_time_unix": time.time()}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if result.returncode and not allow_failure:
            raise RuntimeError(f"{name} failed; see {log}")
        return result.returncode, log

    def train(self, name, config, allow_failure=False):
        run = self.directory / name
        path = self.configs / f"{name}.json"
        if path.exists() and read_json(path) != config:
            raise ValueError(f"Refusing changed cell configuration: {path}")
        if not path.exists():
            atomic_json(path, config)
        if (run / "status.json").exists() and read_json(run / "status.json")["status"] == "complete":
            return 0, None
        arguments = ["experiment.py", "train", "--config", path, "--run-dir", run]
        if run.exists():
            latest = run / "latest_checkpoint.json"
            if not latest.exists():
                if allow_failure and (run / "status.json").exists() and read_json(run / "status.json")["status"] == "failed":
                    return 1, None
                raise ValueError(f"Existing incomplete run has no complete restart; inspect it manually: {run}")
            arguments += ["--resume", read_json(latest)["path"]]
        return self.command(name, arguments, allow_failure)

    def evaluate(self, name, model):
        output = self.directory / f"eval_{name}"
        if (output / "metrics.json").exists():
            metrics = read_json(output / "metrics.json")
            if metrics["status"] != "complete" or metrics["n"] != 11332:
                raise ValueError("Incomplete evaluation cannot count as a result")
            atomic_json(self.directory / f"evaluation_{name}.json", {"path": str(output), "metrics": metrics})
            return
        # Never overwrite an interrupted evaluation. Keep it and use a new attempt.
        if output.exists():
            output = self.directory / f"eval_{name}_attempt_{time.time_ns()}"
        self.command(f"evaluate_{name}", ["experiment.py", "evaluate", "--model", model,
                     "--prepared-dir", PREPARED, "--output-dir", output, "--split", "validation",
                     "--prefix", PREFIX, "--max-source-tokens", "1024", "--max-target-tokens", "64",
                     "--batch-size", "8"])
        metrics = read_json(output / "metrics.json")
        if metrics["status"] != "complete" or metrics["n"] != 11332:
            raise ValueError("Evaluation did not cover full validation")
        atomic_json(self.directory / f"evaluation_{name}.json", {"path": str(output), "metrics": metrics})

    def checkpoint(self, name):
        run = self.directory / name
        status = read_json(run / "status.json")
        if status["status"] != "complete":
            raise ValueError(f"Training is not complete: {name}")
        path = Path(read_json(run / "latest_checkpoint.json")["path"])
        for file, digest in read_json(path / "COMPLETE.json")["files"].items():
            if Path(file).name != file or sha256(path / file) != digest:
                raise ValueError("Checkpoint verification failed")
        return path

    def pilots(self):
        saved = self.directory / "pilot_results.json"
        if saved.exists():
            results = read_json(saved)["results"]
            selected = select_microbatch(results)[0]
            if self.spec.get("microbatch") and selected != self.spec["microbatch"]:
                raise ValueError("Saved pilot selection disagrees with the fixed study microbatch")
            return selected
        # Long training articles stress the fixed 1024/64 tensor shapes; never use validation for fit selection.
        # External SFT weights can have different allocation behavior. Test those
        # actual weights with the full host dataset resident, not the base model.
        pilot_data = PREPARED if self.spec.get("student_init") else self.directory / "pilot_data"
        pilot_student = Path(self.spec.get("student_init", ROOT / "assets/student"))
        if not pilot_data.exists():
            with (PREPARED / "train.jsonl").open(encoding="utf-8") as stream:
                rows = heapq.nlargest(96, (json.loads(line) for line in stream if line.strip()),
                                     key=lambda row: len(row["document"]))
            pilot_data.mkdir()
            path = pilot_data / "train.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest = read_json(PREPARED / "manifest.json")
            manifest["subsets"] = {"train": {"count": 96, "sha256": sha256(path)}}
            atomic_json(pilot_data / "manifest.json", manifest)
            del rows
        results = []
        sizes = (self.spec["microbatch"],) if self.spec.get("microbatch") else (1, 2, 4)
        for branch, fraction in ORDER[:2]:
            for size in sizes:
                name = f"pilot_{branch}_b{size}"
                config = recipe(name, pilot_student, size, fraction=fraction, prepared=pilot_data)
                config.update(max_steps=3, schedule_horizon=3, warmup_steps=0, save_every=3)
                returncode, log = self.train(name, config, allow_failure=True)
                if returncode:
                    message = log.read_text(encoding="utf-8") if log else read_json(self.directory / name / "status.json").get("error", "")
                    if "out of memory" not in message.lower():
                        raise RuntimeError(f"Pilot failed for a reason other than OOM; inspect {log}")
                    results.append({"branch": branch, "microbatch": size, "status": "oom"})
                else:
                    metrics = load_metrics(self.directory / name / "metrics.jsonl")
                    results.append({"branch": branch, "microbatch": size, "status": "complete",
                                    "warmed_update_seconds": statistics.mean(row["update_seconds"] for row in metrics[1:]),
                                    "peak_allocated_bytes": metrics[-1]["peak_allocated_bytes"],
                                    "peak_reserved_bytes": metrics[-1]["peak_reserved_bytes"]})
                atomic_json(self.directory / "pilot_progress.json", {"results": results})
        selected, scores = select_microbatch(results)
        atomic_json(saved, {"results": results, "selected_microbatch": selected,
                            "geometric_mean_update_seconds": scores,
                            "note": "Pilots are discarded; main runs restart from the common SFT asset."})
        return selected

    def execute(self, pilots_only=False):
        if not PREPARED.exists():
            self.command("prepare", ["experiment.py", "prepare", "--output-dir", PREPARED,
                         "--train-size", "204045", "--validation-size", "11332", "--test-size", "64"])
        prepare_data()
        selected = self.pilots()
        self.status("pilots_complete", selected_microbatch=selected)
        if pilots_only:
            return
        # Only common SFT is shared; no GKD cell is initialized from another cell.
        if self.spec.get("student_init"):
            common = Path(self.spec["student_init"])
            metadata = read_json(common / "metadata.json")
            if metadata["config"]["algorithm"] != "sft":
                raise ValueError("Reusable common initialization must be an SFT checkpoint, not a GKD cell")
            for file, digest in read_json(common / "COMPLETE.json")["files"].items():
                if Path(file).name != file or sha256(common / file) != digest:
                    raise ValueError("Reusable SFT checkpoint failed verification")
        else:
            sft = recipe("common_sft", ROOT / "assets/student", selected,
                         self.spec["sft_seconds"], algorithm="sft")
            self.train("common_sft", sft)
            common = self.checkpoint("common_sft")
        frozen = {"path": str(common), "files": read_json(common / "COMPLETE.json")["files"]}
        freeze_path = self.directory / "common_initialization.json"
        if freeze_path.exists() and read_json(freeze_path) != frozen:
            raise ValueError("Common student initialization changed")
        atomic_json(freeze_path, frozen)
        self.evaluate("common_sft", common)
        results = {}
        for name, fraction in ORDER:
            config = recipe(name, common, selected, self.spec["budget_seconds"], fraction=fraction)
            self.train(name, config)
            checkpoint = self.checkpoint(name)
            self.evaluate(name, checkpoint)
            record = read_json(self.directory / f"evaluation_{name}.json")
            status = read_json(self.directory / name / "status.json")
            if status["consumed_seconds"] < self.spec["budget_seconds"]:
                raise ValueError("Update safety ceiling stopped a cell before its compute budget; do not claim budget matching")
            results[name] = {"lambda": fraction, "training": status, "evaluation": record}
            atomic_json(self.directory / "results.json", results)
        for name in ("on_policy", "mixed"):
            a = Path(results[name]["evaluation"]["path"]) / "predictions.jsonl"
            b = Path(results["off_policy"]["evaluation"]["path"]) / "predictions.jsonl"
            output = self.directory / f"{name}_minus_off_policy.json"
            if not output.exists():
                self.command(f"compare_{name}", ["experiment.py", "compare", "--a", a, "--b", b,
                             "--metric", "rouge2", "--bootstrap-samples", "2000", "--output", output])
        self.status("complete", selected_microbatch=selected, results=str(self.directory / "results.json"))


def load_metrics(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "runs/gkd_jordiclive_b60_seed17")
    parser.add_argument("--budget-minutes", type=float, default=60)
    parser.add_argument("--sft-minutes", type=float, default=60)
    parser.add_argument("--student-init", type=Path,
                        help="Reuse a verified common SFT checkpoint across new budget studies; never a GKD cell")
    parser.add_argument("--microbatch", type=int, choices=[1, 2, 4],
                        help="Fix the microbatch for all cells; it must still pass off/on pilots")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pilots-only", action="store_true")
    args = parser.parse_args()
    for value in (args.budget_minutes, args.sft_minutes):
        if not math.isfinite(value) or value <= 0:
            parser.error("Budgets must be finite positive minutes")
    directory = args.study_dir.resolve()
    spec = {"budget_seconds": args.budget_minutes * 60, "sft_seconds": args.sft_minutes * 60,
            "teacher": str(TEACHER), "prepared": str(PREPARED), "seed": 17,
            "source_order": ORDER, "budget_clock": "cumulative synchronized awake update seconds",
            "excludes": ["pilots", "shared SFT", "loading", "checkpoint IO", "evaluation", "OS suspend"],
            "evaluation": "full XSum validation, instruction prefix, greedy BF16 1024/64, batch 8",
            "paper_departures": ["FLAN mixed-data teacher", "budget-scaled common SFT, not 100K updates",
                                 "runtime budget, not 40K GKD updates", "instruction prefix",
                                 "balanced microbatch mixture", "one training seed"]}
    # Normalize tuples now so resume comparisons match JSON round-trips.
    spec = json.loads(json.dumps(spec))
    if args.student_init:
        spec["student_init"] = str(args.student_init.resolve())
        spec["paper_departures"] = [item for item in spec["paper_departures"] if item != "budget-scaled common SFT, not 100K updates"]
        spec["paper_departures"].append("external common SFT checkpoint; see its preparation metadata")
    if args.microbatch:
        spec["microbatch"] = args.microbatch
    print(json.dumps({"study_dir": str(directory), **spec}, indent=2), flush=True)
    if not args.run:
        print("Plan only. Add --run to prepare local data, pilot, train and evaluate.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    spec_path = directory / "study.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError("Study definition changed; use a new directory, never rewrite an existing study")
    if not spec_path.exists():
        atomic_json(spec_path, spec)
    lock = directory / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    study = Study(directory, spec)
    try:
        with keep_awake():
            verified = verify_download(TEACHER)
            provenance = {"manifest_sha256": sha256(TEACHER / MANIFEST), "files": verified["files"]}
            provenance_path = directory / "teacher_provenance.json"
            if provenance_path.exists() and read_json(provenance_path) != provenance:
                raise ValueError("Pinned teacher changed")
            atomic_json(provenance_path, provenance)
            study.execute(args.pilots_only)
    except BaseException as error:
        study.status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
