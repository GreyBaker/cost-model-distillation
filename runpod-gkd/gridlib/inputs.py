"""Read and verify completed SFT assets. No model loads or CUDA calls."""
from __future__ import annotations

from pathlib import Path

from .policy import PROTOCOL, code_signature
from .shared import SFT_CODE, atomic_json, checkpoints, read_json, sha256
from sftlib.resume import record_migration


def separate_work(root, sft_work):
    root, sft_work = Path(root).resolve(), Path(sft_work).resolve()
    if (root.is_relative_to(sft_work) or sft_work.is_relative_to(root)
            or root.is_relative_to(SFT_CODE) or SFT_CODE.is_relative_to(root)):
        raise ValueError("GKD work must be separate from SFT work and SFT source code")


def inspect_pair(sft_work, full=True):
    from sftlib import data, inference
    from sftlib.core import asset_spec
    sft_work = Path(sft_work).resolve()
    pair_path = sft_work / "prepared_pair.json"
    pair = read_json(pair_path)
    if pair.get("status") != "complete" or set(pair.get("roles", {})) != {"student", "teacher"}:
        raise ValueError("Both SFT roles must finish before binding/running GKD")
    paths, configs, hashes = {}, {}, {"prepared_pair": sha256(pair_path)}
    for role, expected_steps in (("student", 100000), ("teacher", 8000)):
        recorded = pair["roles"][role]
        checkpoint = Path(recorded["fp32_checkpoint"]).resolve()
        export = Path(recorded["bf16_export"]).resolve()
        role_root = sft_work / role
        if (not checkpoint.is_relative_to(role_root / "runs")
                or not export.is_relative_to(role_root / "exports")):
            raise ValueError("Pair has foreign/relocated paths; restore the original Pod paths")
        run = read_json(checkpoint.parent.parent / "run.json")
        value = run["config"]
        assets_path = role_root / "assets/manifest.json"
        assets = read_json(assets_path)
        if (assets.get("status") != "complete" or assets.get("assets") != asset_spec(value)
                or run["signatures"]["assets"] != sha256(assets_path)):
            raise ValueError("SFT base/data pins or asset provenance changed")
        metrics = recorded["metrics"]
        if metrics.get("status") != "complete" or metrics.get("n") != 11332 or metrics.get("split") != "validation":
            raise ValueError("Finish both full SFT validation evaluations before GKD")
        if (value["role"] != role or value["max_steps"] != expected_steps
                or any(value[k] != PROTOCOL[k] for k in ("prefix", "max_source_tokens", "max_target_tokens"))):
            raise ValueError("Unexpected SFT model/step budget/prompt/lengths")
        marker = read_json(checkpoint / "COMPLETE.json")
        if marker["step"] != expected_steps or marker["run_id"] != run["run_id"]:
            raise ValueError("SFT checkpoint is incomplete/foreign")
        if full:
            checkpoints.verify(checkpoint, run["run_id"])
            exported = inference.verify_export(export)
            data.verify(role_root, value)
        else:
            exported = read_json(export / "COMPLETE.json")
        provenance = read_json(export / "provenance.json")
        if (exported.get("status") != "complete" or exported.get("role") != role
                or exported.get("step") != expected_steps or provenance["source_run"] != run
                or provenance.get("partial") is not False
                or provenance["source_checkpoint_complete_sha256"] != sha256(checkpoint / "COMPLETE.json")
                or recorded["export_complete_sha256"] != sha256(export / "COMPLETE.json")
                or run["signatures"]["prepared"] != sha256(role_root / "prepared/manifest.json")):
            raise ValueError("SFT export, checkpoint and prepared-data provenance disagree")
        configs[role] = value
        paths[f"{role}_checkpoint"] = str(checkpoint)
        paths[f"{role}_export"] = str(export)
        hashes[role] = {"checkpoint": sha256(checkpoint / "COMPLETE.json"),
                        "export": sha256(export / "COMPLETE.json"),
                        "prepared": sha256(role_root / "prepared/manifest.json")}
    if (pair["student_for_fresh_gkd_training"] != paths["student_checkpoint"]
            or pair["teacher_for_frozen_gkd_scoring"] != paths["teacher_export"]):
        raise ValueError("Pair entry points disagree with role artifacts")
    return {"sft_work": str(sft_work), "paths": paths, "hashes": hashes,
            "sft_configs": configs}


def binding(root, sft_work, settings, create=False, full=True):
    separate_work(root, sft_work)
    definition = {"inputs": inspect_pair(sft_work, full=full), "settings": settings,
                  "protocol": PROTOCOL, "code": code_signature()}
    path = root / "binding.json"
    if path.exists():
        saved = read_json(path)
        if ({k: v for k, v in saved.items() if k != "code"}
                != {k: v for k, v in definition.items() if k != "code"}):
            raise ValueError("Bound SFT inputs/settings changed. Use a new grid directory.")
        if create or full:
            record_migration(root / "code_migrations", {"code": saved["code"]}, {"code": definition["code"]})
    elif create:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(path, definition)
    else:
        raise ValueError("Run bind after both SFT roles finish")
    return definition
