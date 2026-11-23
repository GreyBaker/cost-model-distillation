"""Standalone BF16 CUDA teacher SFT on Runpod/Lightning. Default/plan imports no ML stack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from teacherlib.core import CONFIG, SPEC, RunLock, config, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TEACHER_WORK", "work")),
                        help="Choose a persistent/network volume; not a disposable container disk")
    parser.add_argument("--config", type=Path, default=CONFIG)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Read-only protocol/storage plan (default)")
    prepare = commands.add_parser("prepare", help="Explicit pinned asset download + train/validation tokenization")
    prepare.add_argument("--assets-only", action="store_true")
    prepare.add_argument("--from-assets", type=Path, help="Copy existing experiment-code assets rather than network download")
    commands.add_parser("preflight", help="Verify inputs and inspect CUDA GPU; loads no model")
    train = commands.add_parser("train", help="Full-parameter SFT, or continue this same run")
    train.add_argument("--resume", choices=["latest"], help="Continue latest complete checkpoint; not another run")
    train.add_argument("--session-updates", type=int, help="Pause after N additional updates, e.g. 2 for a pilot")
    train.add_argument("--accept-reconstructed-recipe", action="store_true",
                       help="Acknowledge that batch/LR/schedule are not the recovered GKD teacher-SFT recipe")
    commands.add_parser("status", help="Inspect status/checkpoint paths without importing torch")
    estimate = commands.add_parser("estimate", help="Project 8K-step time/cost from saved pilot measurements")
    estimate.add_argument("--hourly-rate", type=float, required=True, help="Actual GPU USD/hour from launch screen")
    estimate.add_argument("--credits-usd", type=float, default=0, help="Confirmed remaining credit USD value, not credit units")
    estimate.add_argument("--extra-restarts", type=int, default=0,
                          help="Optional future restart allowance for the estimate only; imposes no timer")
    cleanup = commands.add_parser("cleanup-incomplete", help="Preview owned partial checkpoint saves only")
    cleanup.add_argument("--apply", action="store_true", help="Explicitly delete the listed owned incomplete saves")
    release = commands.add_parser("release-base", help="Preview freeing cached base weight after two verified restarts")
    release.add_argument("--apply", action="store_true", help="Explicitly unlink only this work directory's cached base .bin")
    export = commands.add_parser("export", help="Write BF16 Transformers export, verify save/reload parity")
    export.add_argument("--allow-partial", action="store_true", help="Allow labeled pilot export before step 8000")
    evaluate = commands.add_parser("evaluate", help="Greedy validation ROUGE of a completed inference export")
    evaluate.add_argument("--export-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--limit", type=int, default=256, help="Validation examples; 0 means all 11,332")
    verify = commands.add_parser("verify-export", help="Offline hash check of exported teacher; no ML imports")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    value = config(args.config)
    work = args.work_dir.expanduser().resolve()
    run_dir = work / "runs" / value["run_name"]
    command = args.command or "plan"
    if command == "plan":
        spec = read_json(SPEC)
        print(json.dumps({"work_dir": str(work), "run_dir": str(run_dir), "config": value,
                          "base": {k: spec["model"][k] for k in ("repo_id", "revision")},
                          "download_bytes": sum(x["bytes"] for group in spec.values() for x in group["files"]),
                          "effective_batch": value["microbatch_size"] * value["gradient_accumulation_steps"],
                          "training_presentations": value["max_steps"] * value["microbatch_size"] * value["gradient_accumulation_steps"],
                          "hardware_policy": {"requires_cuda_native_bf16": True, "gpu_name_restriction": None,
                                              "advisory_vram_gib": value["gpu_memory_warning_gib"]},
                          "checkpoint_policy": {"every_updates": value["save_every_steps"], "keep_complete": value["keep_checkpoints"],
                                                "also_save": ["pilot pause", "graceful stop", "final update"]},
                          "automatic_time_limit": None, "export_precision": "bfloat16",
                          "storage": "~36-40 GB training steady / ~48-52 GB during staged save, excluding environment; optional explicit release-base after two verified restarts frees ~11.4 GB. Export adds ~5.7 GB after training.",
                          "warning": "Validate free space/quota and GPU fit. SFT batch/LR/schedule are reconstructed, not recovered paper settings.",
                          "operation": "plan only; no model imports/downloads/training"}, indent=2))
        return
    if command == "status":
        from teacherlib.checkpoints import completed
        print(json.dumps({"work_dir": str(work), "status": read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else None,
                          "checkpoints": [str(x) for x in completed(run_dir)] if (run_dir / "run.json").exists() else [],
                          "free_bytes": shutil.disk_usage(work).free if work.exists() else None}, indent=2))
        return
    if command == "verify-export":
        from teacherlib.inference import verify_export
        marker = verify_export(args.directory.resolve())
        print(json.dumps({"status": "verified", "step": marker["step"], "files": len(marker["files"])}, indent=2))
        return
    if command == "estimate":
        from teacherlib.estimates import report
        print(json.dumps(report(run_dir, value, args.hourly_rate, args.credits_usd, args.extra_restarts), indent=2))
        return
    # One writer per work directory; kernel releases the lock even after SIGKILL.
    with RunLock(work / ".teacher.lock"):
        if command == "prepare":
            from teacherlib.assets import acquire
            acquire(work, args.from_assets.resolve() if args.from_assets else None)
            if not args.assets_only:
                from teacherlib.data import prepare
                prepare(work, value)
        elif command == "preflight":
            from teacherlib.assets import verify
            from teacherlib.data import verify as verify_data
            from teacherlib.modeling import cuda_device
            from teacherlib.core import environment
            run = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else None
            verify(work, run["run_id"] if run else None)
            verify_data(work, value)
            import torch
            device = cuda_device(value)
            print(json.dumps({"environment": environment(), "gpu": torch.cuda.get_device_name(device),
                              "gpu_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
                              "free_disk_gib": shutil.disk_usage(work).free / 1024**3}, indent=2))
        elif command == "train":
            if not args.resume and not args.accept_reconstructed_recipe:
                raise ValueError("Read LITERATURE_CHECK.md/DECISIONS.md, then explicitly pass --accept-reconstructed-recipe for a new run")
            from teacherlib.training import train
            train(work, value, args.resume, session_updates=args.session_updates)
        elif command == "cleanup-incomplete":
            from teacherlib.checkpoints import cleanup_incomplete
            print(json.dumps(cleanup_incomplete(run_dir, args.apply), indent=2))
        elif command == "release-base":
            from teacherlib.assets import release_base
            print(json.dumps(release_base(work, run_dir, args.apply), indent=2))
        elif command == "export":
            from teacherlib.inference import export
            export(work, value, allow_partial=args.allow_partial)
        elif command == "evaluate":
            from teacherlib.inference import evaluate
            evaluate(work, value, args.export_dir.resolve(), args.output_dir.resolve(), args.limit)


if __name__ == "__main__":
    main()
