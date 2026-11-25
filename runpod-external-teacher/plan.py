"""Read-only planner for the implemented external_teacher.py workflow."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(scheme, locked):
    require(scheme["status"] == "implemented", "Invalid workflow implementation status")
    require(scheme["default_student"] == "restored", "Our student is the default; alternate mode must be explicit")
    require(scheme["accounting"]["local_sft_updates"] == 0, "This scheme must not perform local SFT")
    require(scheme["validation"]["periodic_greedy_monitor"] is False, "No periodic greedy monitor")
    grid = scheme["grid"]
    require(grid["budgets"] == [500, 1000, 5000, 10000, 50000], "Grid budgets changed")
    require(grid["on_policy_percentages"] == [0, 25, 50, 75, 100], "Grid mixture axis changed")
    require(grid["effective_batch"] == 32, "Grid batch must stay 32, not teacher SFT batch128")
    require(grid["off_policy_point_cost"] == 1 and grid["on_policy_point_cost"] == 2, "Point costs changed")
    require(grid["training_temperature"] == grid["loss_temperature"] == 1, "Temperatures must remain 1")
    require(grid["loss"] == "forward_kl_teacher_to_student", "Grid objective changed")
    require(scheme["validation"]["count"] == locked["evaluation_data"]["counts"]["validation"] == 11332,
            "Validation coverage changed")
    require(locked["evaluation_data"]["counts"]["train"] == 204045, "Training coverage changed")
    for label, source in [*locked["models"].items(), ("data", locked["evaluation_data"])]:
        require(re.fullmatch(r"[a-f0-9]{40}", source["revision"]) is not None,
                f"{label}: immutable commit required, not a branch")
        seen = set()
        for item in source["files"]:
            name = item["path"]
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
                    and ":" not in name and name not in ("", ".") and name not in seen,
                    f"{label}: unsafe or repeated file name")
            seen.add(name)
            require(type(item["bytes"]) is int and item["bytes"] > 0, f"{label}: invalid byte count")
            keys = [key for key in ("sha256", "git_blob_sha1") if key in item]
            require(len(keys) == 1, f"{label}: declare exactly one known source hash")
            key = keys[0]
            require(re.fullmatch(r"[a-f0-9]{" + str(64 if key == "sha256" else 40) + "}", item[key]) is not None,
                    f"{label}: invalid {key}")
        if label != "data":
            require("model.safetensors" in seen and "config.json" in seen and "tokenizer.json" in seen,
                    f"{label}: incomplete model source manifest")
    require(scheme["runtime"]["grid_transformers"] == locked["models"]["teacher"]["saved_transformers_version"],
            "Default native environment must match the public teacher export")
    return scheme, locked


def grid_cells(grid):
    cells = []
    for budget in grid["budgets"]:
        for percent in grid["on_policy_percentages"]:
            fraction = Fraction(percent, 100)
            online, group = fraction.numerator, fraction.denominator
            cost = group + online
            groups = budget // cost
            per_update = grid["effective_batch"] // group
            cells.append({
                "budget_points": budget, "on_policy_percent": percent,
                "points": groups * cost, "examples": groups * group,
                "on_policy_examples": groups * online,
                "off_policy_examples": groups * (group - online),
                "updates": (groups + per_update - 1) // per_update,
                "unspent_points": budget % cost,
            })
    return cells


def build(student="restored"):
    require(student in ("restored", "public"), "Choose restored or public student explicitly")
    scheme, locked = validate(read(ROOT / "scheme.json"), read(ROOT / "models.lock.json"))
    stages = [
        "preflight_owned_one_disk_paths_and_peak_space",
        "create_isolated_native_5_20_2_environment_and_freeze_versions",
    ]
    if student == "restored":
        stages += ["verify_copied_complete_100k_student_read_only",
                   "create_separate_4_57_6_student_reference_environment"]
    stages += ["download_and_verify_locked_teacher_sources"]
    if student == "public":
        stages += ["download_and_verify_locked_public_student_sources"]
    stages += [
        "audit_native_tensor_aliases_scaling_and_loading_diagnostics",
        "audit_token_ids_normalization_and_data_tokenization",
        "verify_or_prepare_full_official_train_and_validation",
    ]
    if student == "restored":
        stages += ["compare_legacy_and_native_student_logits_gradients_and_parameter_aliases"]
    stages += [
        "verify_native_fp32_student_and_bf16_teacher_serialization",
        "full_11332_teacher_and_student_validation_and_paired_difference",
        "disposable_gkd_update_save_reload_resume_smoke_test",
        "review_and_explicitly_approve_quality_and_compatibility_reports",
        "publish_honest_external_pair_manifest_atomically",
        "bind_new_grid_through_required_external_pair_adapter",
        "existing_joint_training_fit_pilot",
        "existing_once_only_student_inference_calibration_and_full_baseline",
        "print_measured_estimate_and_wait_for_grid_approval",
        "run_all_25_cells_independently_from_fixed_fp32_student",
        "audit_hashes_points_predictions_and_download_outputs",
    ]
    cells = grid_cells(scheme["grid"])
    selected = locked["models"]["public_student"] if student == "public" else {
        "kind": "restored_local_sft", "source": scheme["storage"]["restored_student_source"],
        "pin": "verify the complete copied role, original run, checkpoint/export/data/evaluation hashes",
    }
    return {
        "status": "implemented_gpu_validation_required",
        "operation": "read-only plan; no downloads, ML imports, CUDA, training, setup or writes",
        "teacher": {key: locked["models"]["teacher"][key] for key in
                    ("repo_id", "revision", "publisher_reported_sft_steps", "publisher_reported_validation_rouge2")},
        "student_mode": student, "student": selected,
        "local_sft_updates": 0, "periodic_greedy_monitor": False,
        "native_transformers": scheme["runtime"]["grid_transformers"],
        "stages": stages, "grid_cells": cells,
        "grid_totals": {"cells": len(cells), **{
            key: sum(item[key] for item in cells) for key in
            ("points", "examples", "on_policy_examples", "off_policy_examples", "updates", "unspent_points")}},
        "implementation_required": [],
        "warning": "Use external_teacher.py run in the isolated native environment, not the local-SFT pipeline. "
                   "No GPU compatibility/quality claim is made by this plan: the real run enforces validation "
                   "and approval gates before any grid training.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=("restored", "public"), default="restored")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(args.student)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("EXTERNAL TEACHER - READ-ONLY PLAN, NOT EXECUTION")
    print(f"Teacher: {result['teacher']['repo_id']} @ {result['teacher']['revision']}")
    print(f"Student: {result['student_mode']} | local SFT updates: 0 | periodic monitor: off")
    print(f"Proposed native Transformers: {result['native_transformers']}")
    for index, stage in enumerate(result["stages"], 1):
        print(f"{index:2}. {stage}")
    print("Grid totals:", json.dumps(result["grid_totals"]))
    print("Operational entry point: external_teacher.py run (explicit opt-in after setup).")
    print(result["warning"])


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
