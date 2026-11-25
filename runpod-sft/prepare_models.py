"""Standalone Runpod XSum SFT: LM-adapted T5-small 100K, then T5-XL 8K. Plan by default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sftlib.core import (CONFIGS, SPEC, RunLock, asset_spec, atomic_json, check_teacher_work, config,
                          configure_local_storage, read_json, recipe_evidence)
from sftlib.console import output as display


def plan(root, config_dir):
    from sftlib.batching import candidates, PILOT_UPDATES, RESERVED_FRACTION_LIMIT
    roles = {}
    for role in ("student", "teacher"):
        value = config(role, config_dir)
        spec = asset_spec(value)
        batch = value["microbatch_size"] * value["gradient_accumulation_steps"]
        roles[role] = {"config": value, "base": {name: spec["model"][name] for name in ("repo_id", "revision")},
                        "work": str(root / role), "effective_batch": batch,
                        "recipe_evidence": recipe_evidence(value),
                        "example_presentations": value["max_steps"] * batch,
                        "sft_microbatch_pilot": {
                            "partitions_largest_first": [f"{size}x{batch // size}" for size in candidates(value)],
                            "discarded_updates_per_candidate": PILOT_UPDATES,
                            "reload_updates": 1, "reserved_fraction_limit": RESERVED_FRACTION_LIMIT,
                            "selection": "largest passing full-shape train/save/reload microbatch; once per role"},
                        "greedy_monitoring": ({"every_steps": value["rouge_every_steps"],
                            "examples": value["rouge_examples"], "advisory_review_step": value["review_step"],
                            "retention": "latest two + pinned subset-best; final 8K exported for grid"}
                            if role == "teacher" else None),
                       "download_bytes": sum(item["bytes"] for group in spec.values() for item in group["files"])}
    return {"operation": "plan only; no ML imports, downloads, CUDA calls or training", "order": ["student", "teacher"],
            "roles": roles, "automatic_time_limit": None, "export": "separate BF16 safetensors with serialization parity probes",
            "hardware": "one native-BF16 CUDA GPU; planned 80 GB-class A100/H100; no name restriction",
            "storage": "One disposable on-Pod disk for source, venv, caches, temp, data and outputs; "
                       "target 150 GB available for SFT plus the later grid; no separate disk or persistence assumed",
             "evidence_boundary": "Student recipe unchanged. Teacher adapts public T5X XL/XSum: batch128, "
                                  "8K updates, parameter-scaled Adafactor 5e-4, warmup800, cooldown6K-8K, "
                                  "z-loss1e-4 and fixed divisor233472 at 1024/64. Not exact GKD-author SFT."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("SFT_WORK", "work")),
                        help="Directory on the short-lived on-Pod disk; source pod_env.sh to set SFT_WORK")
    parser.add_argument("--config-dir", type=Path, default=CONFIGS)
    parser.add_argument("--json", action="store_true", help="JSON for reports instead of readable text")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Read-only plan (default)")
    prepare = commands.add_parser("prepare", help="Explicit pinned downloads and local train/validation tokenization")
    prepare.add_argument("--role", choices=["student", "teacher", "both"], default="both")
    prepare.add_argument("--assets-only", action="store_true")
    prepare.add_argument("--from-assets", type=Path, help="Import existing pinned student/teacher/xsum asset subdirectories instead of downloading")
    preflight = commands.add_parser("preflight", help="ON THE POD: verify inputs and inspect CUDA/BF16; no model load")
    preflight.add_argument("--role", choices=["student", "teacher"], required=True)
    train = commands.add_parser("train", help="Train one role, optionally with an additional-update pilot cap")
    train.add_argument("--role", choices=["student", "teacher"], required=True)
    train.add_argument("--resume", choices=["latest"])
    train.add_argument("--session-updates", type=int)
    train.add_argument("--accept-reconstructed-recipe", action="store_true")
    pilot = commands.add_parser("pilot", help="ON POD: once-only largest-fit SFT microbatch pilot; discarded weights")
    pilot.add_argument("--role", choices=["student", "teacher"], required=True)
    for name in ("_sft_runtime", "_sft_pilot", "_train"):
        worker = commands.add_parser(name, help=argparse.SUPPRESS)
        worker.add_argument("--role", choices=["student", "teacher"], required=True)
        worker.add_argument("--output", type=Path)
        worker.add_argument("--microbatch", type=int)
        worker.add_argument("--resume", choices=["latest"])
        worker.add_argument("--session-updates", type=int)
        worker.add_argument("--attempt-result", type=Path)
    export = commands.add_parser("export", help="ON THE POD: verified BF16 export of a complete role")
    export.add_argument("--role", choices=["student", "teacher"], required=True)
    export.add_argument("--allow-partial", action="store_true", help="Labeled pilot export, never a final prepared pair")
    evaluate = commands.add_parser("evaluate", help="ON THE POD: greedy full validation ROUGE of a final BF16 role export")
    evaluate.add_argument("--role", choices=["student", "teacher"], required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--limit", type=int, default=0, help="0 = all 11,332 validation examples")
    run = commands.add_parser("run", help="ON THE POD: serial student train/export/eval, then teacher train/export/eval")
    run.add_argument("--accept-reconstructed-recipe", action="store_true")
    reuse = run.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-student", action="store_true",
                     help="Verify existing complete student role read-only; skip its train/export/evaluation workers")
    reuse.add_argument("--reuse-sft", action="store_true",
                       help="Verify both restored complete roles and rebuild the pair manifest; no SFT workers")
    commands.add_parser("verify-student", help="Read-only hashes/provenance/full-evaluation check of a restored SFT student")
    commands.add_parser("status", help="Read-only status of both roles; no ML imports")
    estimate = commands.add_parser("estimate", help="Read-only pilot-based remaining/full training time and compute USD")
    estimate.add_argument("--role", choices=["student", "teacher"], required=True)
    estimate.add_argument("--hourly-rate", type=float, required=True)
    verify = commands.add_parser("verify-export", help="Offline export hash verification; no ML imports")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    root, config_dir = args.work_dir.expanduser().resolve(), args.config_dir.expanduser().resolve()
    operation = args.command or "plan"
    if operation == "plan":
        display(plan(root, config_dir), "Paired SFT plan", args.json)
        return
    if operation == "status":
        from sftlib.checkpoints import completed
        result = {}
        for role in ("student", "teacher"):
            value = config(role, config_dir)
            run_dir = root / role / "runs" / value["run_name"]
            result[role] = {"status": read_json(run_dir / "status.json") if (run_dir / "status.json").exists() else None,
                            "checkpoints": [str(path) for path in completed(run_dir)]}
        result["sequence"] = read_json(root / "status.json") if (root / "status.json").exists() else None
        display(result, "SFT status", args.json)
        return
    if operation == "verify-student":
        from sftlib.sequence import verified_student
        display(verified_student(root, config_dir), "Reusable SFT student", args.json)
        return
    if operation == "verify-export":
        from sftlib.inference import verify_export
        marker = verify_export(args.directory.resolve())
        display({"status": "verified", "role": marker["role"], "step": marker["step"]},
               "Export integrity", args.json)
        return
    if operation == "estimate":
        from sftlib.estimates import report
        value = config(args.role, config_dir)
        display(report(root / args.role, value, args.hourly_rate), "SFT runtime estimate", args.json)
        return
    if any(path.name.lower() == "results_and_temp" for path in (root, *root.parents)):
        raise ValueError("results_and_temp is immutable; restore copies into a separate SFT work directory")
    if operation == "run":
        if not args.accept_reconstructed_recipe and not args.reuse_sft:
            raise ValueError("Read DECISIONS.md, then pass --accept-reconstructed-recipe")
        configure_local_storage(root)
        from sftlib.sequence import run
        run(root, config_dir, reuse_student=args.reuse_student, reuse_sft=args.reuse_sft)
        return
    if operation in ("train", "pilot"):
        if operation == "train" and not args.resume and not args.accept_reconstructed_recipe:
            raise ValueError("Read DECISIONS.md, then pass --accept-reconstructed-recipe")
        configure_local_storage(root)
        from sftlib.batching import run
        # Supervisor and GPU workers must never contend for the same lock.
        with RunLock(root / ".training-controller.lock"):
            run(root, config_dir, config(args.role, config_dir),
                resume=args.resume if operation == "train" else None,
                session_updates=args.session_updates if operation == "train" else None,
                pilot_only=operation == "pilot")
        return
    if operation == "prepare" and args.role == "both":
        configure_local_storage(root)
        import subprocess
        import sys
        from sftlib.core import ROOT
        # Distinct supervisor lock: child role workers acquire the global writer
        # lock themselves. Never hold that same lock in the waiting parent.
        with RunLock(root / ".prepare-sequence.lock"):
            for role in ("student", "teacher"):
                command = [sys.executable, str(ROOT / "prepare_models.py"), "--work-dir", str(root),
                           "--config-dir", str(config_dir), "prepare", "--role", role]
                if args.assets_only:
                    command.append("--assets-only")
                if args.from_assets:
                    command += ["--from-assets", str(args.from_assets.resolve())]
                subprocess.run(command, check=True)
        return
    # Only explicit operational subcommands write, import ML or access the network.
    configure_local_storage(root)
    with RunLock(root / ".writer.lock"):
        if operation == "prepare":
            from sftlib import assets
            value = config(args.role, config_dir)
            work = root / args.role
            check_teacher_work(work, value)
            assets.acquire(work, value, args.from_assets.resolve() if args.from_assets else None)
            if not args.assets_only:
                from sftlib.data import prepare
                prepare(work, value)
            return
        value = config(args.role, config_dir)
        work = root / args.role
        if operation == "_sft_runtime":
            from sftlib.batching import runtime_worker
            if args.output is None or args.output.resolve() != work / "pilot_runtime.json":
                raise ValueError("Explicit role-local pilot runtime output required")
            runtime_worker(value, args.output.resolve())
        elif operation == "_sft_pilot":
            from sftlib.batching import pilot_worker, candidates
            if (args.microbatch not in candidates(value) or args.output is None
                    or not args.output.resolve().is_relative_to(work / "microbatch_pilots")):
                raise ValueError("Explicit role-local SFT pilot candidate/output required")
            pilot_worker(work, value, args.microbatch, args.output.resolve())
        elif operation == "_train":
            from sftlib.batching import OOM_EXIT_CODE, active_microbatch, partition
            if (args.attempt_result is None
                    or not args.attempt_result.resolve().is_relative_to(work / "attempts")):
                raise ValueError("Explicit role-local fresh attempt report required")
            actual = active_microbatch(work, value, read_json(work / "pilot_runtime.json"))
            if args.microbatch != actual:
                raise ValueError("SFT worker batch differs from saved pilot/recovery policy")
            import torch
            from sftlib.training import train
            try:
                train(work, partition(value, actual), args.resume, args.session_updates)
            except torch.cuda.OutOfMemoryError as error:
                atomic_json(args.attempt_result.resolve(), {
                    "status": "cuda_oom", "operation": "train", "role": args.role,
                    "microbatch": actual, "error": str(error)})
                raise SystemExit(OOM_EXIT_CODE)
        elif operation == "preflight":
            from sftlib import assets, data, modeling
            from sftlib.core import environment
            assets.verify(work, value)
            data.verify(work, value)
            import torch
            device = modeling.cuda_device(value)
            free, total = torch.cuda.mem_get_info(device)
            display({"role": args.role, "environment": environment(), "gpu": torch.cuda.get_device_name(device),
                    "free_gib": free / 1024**3, "total_gib": total / 1024**3}, "GPU preflight", args.json)
        elif operation == "export":
            from sftlib.inference import export
            export(work, value, args.allow_partial)
        elif operation == "evaluate":
            from sftlib.inference import evaluate
            output = args.output_dir.resolve() if args.output_dir else work / "evaluation"
            metrics = evaluate(work, value, output, args.limit)
            atomic_json(work / "latest_evaluation.json", {"path": str(output), "n": metrics["n"]})


if __name__ == "__main__":
    main()
