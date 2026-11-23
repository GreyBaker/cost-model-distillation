"""Queue fresh 45/30-minute GKD studies from the existing common SFT asset."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from gkdproto.common import ROOT, atomic_json, environment, read_json, sha256
from run_gkd_study import PREFIX, Study, keep_awake
from run_sft100k_followup import await_study, validate_completed_study


def budget_directory(minutes):
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Budgets must be finite positive minutes")
    label = format(minutes, "g").replace(".", "p")
    return ROOT / "runs" / f"gkd_jordiclive_b{label}_mb1_seed17"


def study_command(minutes, directory, common):
    return ["run_gkd_study.py", "--run", "--study-dir", str(directory),
            "--student-init", str(common), "--microbatch", "1", "--budget-minutes", str(minutes)]


def validate_initialization(frozen):
    common = Path(frozen["path"])
    complete = read_json(common / "COMPLETE.json")
    if complete["files"] != frozen["files"]:
        raise ValueError("Common SFT completion manifest changed")
    metadata = read_json(common / "metadata.json")
    if metadata["config"]["algorithm"] != "sft":
        raise ValueError("New budget cells must start from common SFT, not another distilled cell")
    for name, digest in frozen["files"].items():
        if Path(name).name != name or sha256(common / name) != digest:
            raise ValueError("Common SFT file hash changed")
    return common


def validate_baseline(directory, common):
    metrics = read_json(directory / "metrics.json")
    settings = read_json(directory / "settings.json")
    if (metrics["status"] != "complete" or metrics["n"] != 11332 or metrics["split"] != "validation"
            or Path(settings["model"]) != common or settings["prefix"] != PREFIX
            or (settings["max_source_tokens"], settings["max_target_tokens"]) != (1024, 64)
            or settings["precision"] != "bfloat16" or settings["decoding"] != "greedy, beam=1"
            or sha256(directory / "predictions.jsonl") != metrics["predictions_sha256"]):
        raise ValueError("Shared baseline is incomplete or uses a different initialization/protocol")
    return {"metrics_sha256": sha256(directory / "metrics.json"),
            "settings_sha256": sha256(directory / "settings.json"),
            "predictions_sha256": metrics["predictions_sha256"]}


def reuse_baseline(source, study, common):
    expected = validate_baseline(source, common)
    output = study / "eval_common_sft"
    if output.exists():
        if validate_baseline(output, common) != expected:
            raise ValueError("Existing baseline differs; refusing to overwrite it")
        return
    study.mkdir(parents=True, exist_ok=True)
    temporary = study / f".baseline-reuse-{uuid.uuid4().hex}.incomplete"
    shutil.copytree(source, temporary)
    if validate_baseline(temporary, common) != expected:
        raise ValueError("Baseline copy failed verification")
    atomic_json(temporary / "reuse_provenance.json", {"source": str(source), **expected,
                                                     "note": "Same frozen SFT baseline reused; not a new evaluation or training run."})
    os.replace(temporary, output)


def code_signature():
    paths = [*sorted((ROOT / "gkdproto").glob("*.py")), ROOT / "run_gkd_study.py",
             ROOT / "run_sft100k_followup.py", Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def summarize(directories, common):
    rows = []
    for directory in directories:
        frozen = read_json(directory / "common_initialization.json")
        if Path(frozen["path"]) != common:
            raise ValueError("Budget study changed the common student")
        spec = read_json(directory / "study.json")
        results = read_json(directory / "results.json")
        for name in ("off_policy", "on_policy", "mixed"):
            training = results[name]["training"]
            rows.append({"study": str(directory), "budget_minutes": spec["budget_seconds"] / 60,
                         "source": name, "updates": training["step"],
                         "consumed_training_seconds": training["consumed_seconds"],
                         "counts": training["counts"], "scores": results[name]["evaluation"]["metrics"]["scores"],
                         "evaluation": results[name]["evaluation"]["path"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-study", type=Path, default=ROOT / "runs/gkd_jordiclive_b60_mb1_seed17")
    parser.add_argument("--queue-dir", type=Path, default=ROOT / "runs/gkd_jordiclive_budget_ladder_seed17")
    parser.add_argument("--budgets", type=float, nargs="+", default=[45, 30])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if len(args.budgets) != len(set(args.budgets)):
        parser.error("Each budget must be unique")
    try:
        destinations = [budget_directory(value) for value in args.budgets]
    except ValueError as error:
        parser.error(str(error))
    previous = args.after_study.resolve()
    directory = args.queue_dir.resolve()
    if len(set([previous, directory, *destinations])) != 2 + len(destinations):
        parser.error("Existing, queue and new budget study directories must differ")
    frozen = read_json(previous / "common_initialization.json")
    common = Path(frozen["path"])
    spec = {"after_study": str(previous), "budgets_minutes": args.budgets,
            "study_directories": [str(path) for path in destinations],
            "common_initialization": frozen, "microbatch": 1, "effective_batch": 32,
            "sources": ["off_policy", "on_policy", "mixed"], "new_sft": False,
            "new_preparation": "None. The 100K SFT follow-up was cancelled by the user.",
            "environment": environment(), "code_sha256": code_signature()}
    print(json.dumps(spec, indent=2), flush=True)
    if not args.run:
        print("Plan only. --run --wait queues serial budget studies after the active 60-minute study.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / "queue.json"
    if plan.exists():
        if read_json(plan) != spec:
            raise ValueError("Budget queue/config/code/software changed; inspect rather than silently migrate")
    elif any(directory.iterdir()):
        raise FileExistsError("Refusing an unrecognized nonempty queue directory")
    else:
        atomic_json(plan, spec)
    lock = directory / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    controller = Study(directory, spec)
    try:
        with keep_awake():
            controller.status("queued", after_study=str(previous), budgets_minutes=args.budgets)
            await_study(previous, args.wait)
            if code_signature() != spec["code_sha256"] or environment() != spec["environment"]:
                raise ValueError("Code/software changed while queued; inspect before running")
            common = validate_initialization(frozen)
            baseline = Path(read_json(previous / "evaluation_common_sft.json")["path"])
            validate_baseline(baseline, common)
            for minutes, destination in zip(args.budgets, destinations):
                reuse_baseline(baseline, destination, common)
                controller.command(f"budget_{format(minutes, 'g')}_minutes", study_command(minutes, destination, common))
                validate_completed_study(destination)
                if read_json(destination / "common_initialization.json") != frozen:
                    raise ValueError("A new budget study did not preserve the exact frozen initialization")
                atomic_json(directory / "results.json", {"rows": summarize([previous, *destinations[:destinations.index(destination) + 1]], common),
                                                          "note": "Independent runs; common SFT/evaluation reused. Runtime proxy, not FLOP equality."})
            controller.status("complete", budgets_minutes=args.budgets, results=str(directory / "results.json"))
    except BaseException as error:
        controller.status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
