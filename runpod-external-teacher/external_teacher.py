"""Explicit external-teacher preparation/validation/grid runner. Plan by default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-disk", type=Path, default=os.environ.get("POD_DISK"))
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("EXTERNAL_WORK", "external-work")))
    parser.add_argument("--grid-work", type=Path, default=Path(os.environ.get("GKD_WORK", "external-grid")))
    parser.add_argument("--student", choices=("restored", "public"), default="restored")
    parser.add_argument("--restored-root", type=Path, default=os.environ.get("RESTORED_SFT"))
    parser.add_argument("--reference-python", type=Path, default=ROOT / ".venv-reference/bin/python")
    parser.add_argument("--settings", type=Path, default=ROOT.parent / "runpod-gkd/settings.json")
    parser.add_argument("--min-teacher-r2", type=float, default=20.)
    parser.add_argument("--min-advantage", type=float, default=1.)
    parser.add_argument("--min-ci-lower", type=float, default=0.)
    parser.add_argument("--minimum-free-gib", type=float, default=100.)
    parser.add_argument("--json", action="store_true")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Read-only; no ML/network/GPU work")
    execute = commands.add_parser("run", help="ON POD: pinned import, validation, approval and normal grid")
    execute.add_argument("--accept-pair", action="store_true", help="Approve reviewed quality/override and passed smoke evidence")
    execute.add_argument("--override-quality", metavar="REASON",
                         help="Accept a completed quality-threshold failure with an audited reason; never bypass integrity/smoke checks")
    execute.add_argument("--accept-grid-estimate", action="store_true")
    commands.add_parser("verify", help="Read-only full byte/provenance verification of an approved pair")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--stage", required=True, choices=[
        "native-runtime", "reference-runtime", "acquire", "corpus", "legacy-tokenizer", "tokenizers", "prepared", "legacy-probe",
        "student", "teacher", "student-validation", "teacher-validation", "quality", "smoke"])
    worker.add_argument("--batch-size", type=int)
    worker.add_argument("--attempt-result", type=Path, required=True)
    args = parser.parse_args()
    if args.command in (None, "plan"):
        from plan import build
        result = build(args.student)
        print(json.dumps(result, indent=2) if args.json else
              "\n".join([result["status"], *result["stages"], result["warning"]]))
        return
    if args.command == "verify":
        from external_workflow.pair import inspect
        result = inspect(args.work_dir.expanduser().resolve() / "pair")
        print(json.dumps(result, indent=2))
        return
    if args.command == "run":
        if args.pod_disk is None:
            parser.error("Set POD_DISK or --pod-disk")
        from external_workflow.controller import run
        args.work_dir = args.work_dir.expanduser().resolve()
        args.grid_work = args.grid_work.expanduser().resolve()
        if args.restored_root:
            args.restored_root = args.restored_root.expanduser().resolve()
        run(args)
        return
    from external_workflow.common import (OOM_EXIT, RunLock, atomic_json, digest,
        operational_paths, read_json)
    work = args.work_dir.expanduser().resolve()
    if os.environ.get("POD_DISK") is None:
        parser.error("Worker needs POD_DISK")
    run = read_json(work / "run.json")
    operational_paths(os.environ["POD_DISK"], work, run["definition"]["grid_work"])
    report = args.attempt_result.expanduser().resolve()
    if not report.is_relative_to(work / "attempts") or report.exists():
        raise ValueError("Explicit fresh role-local worker attempt report required")
    from sftlib.core import configure_local_storage
    configure_local_storage(work)
    with RunLock(work / ".worker.lock"):
        try:
            stage = args.stage
            if stage in ("native-runtime", "reference-runtime"):
                from external_workflow.common import runtime
                runtime(work, reference=stage == "reference-runtime", gpu=True)
            elif stage == "acquire":
                from external_workflow.acquire import acquire
                acquire(work)
            elif stage == "corpus":
                from external_workflow.data import corpus
                corpus(work)
            elif stage in ("legacy-tokenizer", "tokenizers"):
                from external_workflow.data import token_audit
                token_audit(work, reference=stage == "legacy-tokenizer")
            elif stage == "prepared":
                from external_workflow.data import prepare
                prepare(work)
            elif stage == "legacy-probe":
                from external_workflow.models import legacy_probe
                legacy_probe(work)
            elif stage in ("student", "teacher"):
                from external_workflow.models import export
                export(work, stage)
            elif stage.endswith("-validation"):
                from external_workflow.evaluation import evaluate
                evaluate(work, stage.split("-")[0], args.batch_size)
            elif stage == "quality":
                from external_workflow.evaluation import quality
                quality(work)
            elif stage == "smoke":
                from external_workflow.smoke import run as smoke
                smoke(work)
        except Exception as error:
            torch = sys.modules.get("torch")
            if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
                atomic_json(report, {"status": "cuda_oom", "stage": args.stage,
                    "batch_size": args.batch_size, "definition_sha256": digest(run["definition"]), "error": str(error)})
                raise SystemExit(OOM_EXIT)
            raise


if __name__ == "__main__":
    main()
