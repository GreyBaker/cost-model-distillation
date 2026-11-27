"""Strict external-pair verification; no ML/model deserialization."""
from __future__ import annotations

from pathlib import Path
import math
import time

from .acquire import matches
from .common import (PAIR_KIND, PROFILE, ROOT, atomic_json, code, digest, environment, identity,
                     owned, pair_root, read_json, request, sha256, verify)
from gridlib.policy import PROTOCOL
from . import quality_policy


def expected_artifacts(definition, quality_overridden=False):
    names = {
        "teacher_source": "sources/teacher", "xsum_source": "sources/xsum",
        "tokenizers": "reports/tokenizers", "corpus": "reports/corpus",
        "prepared": "student/prepared", "student": "models/student_fp32", "teacher": "models/teacher_bf16",
        "student_validation": "validation/student", "teacher_validation": "validation/teacher",
        "quality": "reports/quality", "smoke": "reports/smoke",
    }
    if definition["student"] == "public":
        names["public_student_source"] = "sources/public_student"
    else:
        names.update(legacy_tokenizer="reports/legacy_tokenizer", legacy_probe="reports/legacy_probe",
                     restored_source="reports/restored_source")
    if quality_overridden:
        names["quality_override"] = quality_policy.RELATIVE
    return names


def stage_identity(definition, stage, **extra):
    return {"definition_sha256": digest(definition), "stage": stage, **extra}


def inspect(root, full=True, check_runtime=True, *, _candidate=None):
    root = Path(root).resolve()
    if any(p.name.lower() == "results_and_temp" for p in (root, *root.parents)):
        raise ValueError("Do not bind original uploaded results as external operational inputs")
    pair = read_json(root / "prepared_pair.json") if _candidate is None else _candidate
    definition = pair["definition"]
    if (pair.get("kind") != PAIR_KIND or pair.get("status") != "complete"
            or pair.get("protocol") != PROTOCOL or pair.get("local_sft_updates") != 0
            or definition.get("student") not in ("restored", "public")):
        raise ValueError("Incomplete/foreign external-pair schema or protocol")
    if set(definition.get("thresholds", {})) != {"teacher_r2", "advantage", "ci_lower"} or any(
            type(x) not in (float, int) or not math.isfinite(x) for x in definition["thresholds"].values()):
        raise ValueError("Invalid frozen quality criteria")
    lock_path = owned(root, "models.lock.json")
    if sha256(lock_path) != pair["models_lock_sha256"] or pair["models_lock_sha256"] != definition["models_lock_sha256"]:
        raise ValueError("Frozen external source lock changed")
    locked = read_json(lock_path)
    quality_overridden = "quality_override" in pair["artifacts"]
    expected = expected_artifacts(definition, quality_overridden)
    if set(pair["artifacts"]) != set(expected):
        raise ValueError("External pair lacks required evidence")
    markers = {}
    for name, relative in expected.items():
        entry = pair["artifacts"][name]
        if entry.get("path") != relative:
            raise ValueError("External artifact path changed/escaped")
        path = owned(root, relative)
        if sha256(path / "COMPLETE.json") != entry["complete_sha256"]:
            raise ValueError("External artifact completion changed")
        markers[name] = verify(path, full=full)
        if markers[name]["identity"]["definition_sha256"] != digest(definition):
            raise ValueError("Mixed external experiment evidence")
    source_labels = ["teacher", "xsum"] + (["public_student"] if definition["student"] == "public" else [])
    for label in source_labels:
        spec = locked["evaluation_data"] if label == "xsum" else locked["models"][label]
        key = label + "_source" if label != "public_student" else "public_student_source"
        if markers[key]["identity"] != stage_identity(definition, "source", label=label, source=spec):
            raise ValueError("Source identity differs from frozen release lock")
        if full:
            for item in spec["files"]:
                if not matches(owned(root / "sources" / label, item["path"]), item):
                    raise ValueError("Source bytes no longer match published immutable revision")
    for role in ("student", "teacher"):
        if markers[role]["identity"] != stage_identity(definition, "model", role=role):
            raise ValueError("Imported model identity changed")
        model_dir = owned(root, expected[role])
        cfg = read_json(model_dir / "config.json")
        if cfg.get("external_pair_profile") != PROFILE or cfg.get("scale_decoder_outputs") is not False:
            raise ValueError("Native model profile/logit scaling changed")
        if (cfg.get("vocab_size") != 32128 or cfg.get("decoder_start_token_id") != 0
                or cfg.get("pad_token_id") != 0 or cfg.get("eos_token_id") != 1):
            raise ValueError("External model dimensions/special IDs changed")
        provenance = read_json(model_dir / "provenance.json")
        if (provenance.get("local_sft_updates") != 0 or provenance.get("profile") != PROFILE
                or provenance.get("runtime") != pair["runtime"]):
            raise ValueError("External source provenance/runtime changed")
        restored = role == "student" and definition["student"] == "restored"
        if (provenance.get("origin") != ("restored-local-sft" if restored else "external-public-sft")
                or provenance.get("historical_steps") != (100000 if role == "student" else 8000)
                or provenance.get("historical_steps_status") != (
                    "verified-local-completion" if restored else "publisher-reported")):
            raise ValueError("External import history must not masquerade as locally measured SFT")
        inv = provenance.get("inventory", {})
        parity = provenance.get("reload_parity", {})
        if (inv.get("scale_decoder_outputs") is not False or not inv.get("shapes")
                or type(inv.get("unique_parameters")) is not int or inv["unique_parameters"] < 1
                or "parameter_aliases" not in inv or "storage_aliases" not in inv
                or parity.get("rtol") != 1e-4 or parity.get("atol") != 1e-4
                or not math.isfinite(parity.get("max_abs_error", float("nan")))):
            raise ValueError("Missing model alias/scaling/probability-reload evidence")
        if role == "teacher" and not provenance.get("fp32_bf16_drift"):
            raise ValueError("Missing teacher probability-drift report")
        origin_sha = (definition["restored_checkpoint_complete_sha256"] if role == "student" and
                      definition["student"] == "restored" else pair["artifacts"][
                          "public_student_source" if role == "student" else "teacher_source"]["complete_sha256"])
        if provenance.get("source_complete_sha256") != origin_sha:
            raise ValueError("Imported model no longer refers to its verified source")
        if (markers[role].get("role") != role
                or markers[role].get("precision") != ("float32" if role == "student" else "bfloat16")):
            raise ValueError("Incorrect common-initialization/teacher precision")
        if (model_dir / "training_state.pt").exists():
            raise ValueError("Imported models must not pretend to have local optimizer/RNG state")
        validation_name = role + "_validation"
        wanted = stage_identity(definition, "validation", role=role,
            model_complete_sha256=pair["artifacts"][role]["complete_sha256"],
            data_complete_sha256=pair["artifacts"]["prepared"]["complete_sha256"], protocol=PROTOCOL)
        if markers[validation_name]["identity"] != wanted:
            raise ValueError("Validation is not for the actual grid model/data")
        metrics = read_json(owned(root, expected[validation_name]) / "metrics.json")
        if (metrics.get("status") != "complete" or metrics.get("n") != 11332
                or metrics.get("split") != "validation" or metrics.get("precision") != "bfloat16"
                or metrics.get("runtime") != pair["runtime"]):
            raise ValueError("Missing full native BF16 validation")
    data = read_json(owned(root, expected["prepared"]) / "data.json")
    if data.get("counts") != {"train": 204045, "validation": 11332} or data.get("protocol") != PROTOCOL:
        raise ValueError("External prepared data coverage/protocol changed")
    if (data.get("source_complete_sha256") != pair["artifacts"]["xsum_source"]["complete_sha256"]
            or data.get("tokenizer_report_sha256") != pair["artifacts"]["tokenizers"]["complete_sha256"]
            or markers["prepared"]["identity"] != stage_identity(definition, "prepared")):
        raise ValueError("Data is not bound to the official source and audited canonical tokenizer")
    for name, stage in (("tokenizers", "tokenizers"), ("corpus", "corpus")):
        if markers[name]["identity"] != stage_identity(definition, stage):
            raise ValueError("Missing native tokenizer/corpus evidence")
    tokens = read_json(root / expected["tokenizers"] / "report.json")
    compare_keys = ("vocab_sha256", "encoding_sha256", "decoding_sha256", "special_ids", "counts")
    for role in ("student", "teacher"):
        report = tokens[role]
        if (report.get("counts") != {"edge": 8, "validation": 11332, "train": 2048}
                or report.get("special_ids") != {"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2}
                or any(not isinstance(report.get(key), str) or len(report[key]) != 64
                       for key in ("vocab_sha256", "encoding_sha256", "decoding_sha256"))):
            raise ValueError("Incomplete tokenizer mapping/preprocessing evidence")
        entries, maximum = report.get("vocabulary_entries"), report.get("max_token_id")
        if (type(entries) is not int or type(maximum) is not int
                or not 0 < entries <= maximum + 1 <= 32128
                or report.get("padded_output_rows") != 32127 - maximum):
            raise ValueError("Missing tokenizer/output vocabulary and padded-row accounting")
    if any(tokens["student"][key] != tokens["teacher"][key] for key in compare_keys):
        raise ValueError("Student/teacher tokenization differs")
    if definition["student"] == "restored":
        for name, stage in (("legacy_tokenizer", "legacy_tokenizer"), ("legacy_probe", "legacy_probe")):
            if markers[name]["identity"] != stage_identity(definition, stage):
                raise ValueError("Restored student lacks its cross-version comparison")
        if markers["restored_source"]["identity"] != stage_identity(
                definition, "restored_source", checkpoint_complete_sha256=definition["restored_checkpoint_complete_sha256"]):
            raise ValueError("Restored student source completion changed")
        student_provenance = read_json(root / expected["student"] / "provenance.json")
        legacy = read_json(root / expected["legacy_probe"] / "report.json")
        legacy_tokens = read_json(root / expected["legacy_tokenizer"] / "report.json")["student"]
        if any(tokens["student"][key] != legacy_tokens[key] for key in compare_keys):
            raise ValueError("Restored tokenizer changed across native migration")
        migration = student_provenance.get("migration", {})
        if (legacy.get("source_complete_sha256") != definition["restored_checkpoint_complete_sha256"]
                or legacy.get("inventory") != student_provenance["inventory"]
                or migration.get("rtol") != 1e-4 or migration.get("atol") != 1e-4
                or not math.isfinite(migration.get("max_abs_error", float("nan")))
                or not (root / expected["legacy_probe"] / "probe.safetensors").is_file()):
            raise ValueError("Restored student did not pass native probability/gradient migration")
    quality_identity = stage_identity(definition, "quality", validation={
        role: pair["artifacts"][role + "_validation"]["complete_sha256"] for role in ("teacher", "student")})
    if markers["quality"]["identity"] != quality_identity:
        raise ValueError("Quality gate is not for this pair's actual evaluations")
    quality = read_json(root / expected["quality"] / "report.json")
    failed = quality_policy.failed_gates(quality, definition["thresholds"])
    override = None
    if failed:
        if not quality_overridden:
            raise ValueError("External teacher quality gate failed; an explicit quality override is required")
        override = quality_policy.current_override(root, definition, full=full)
    elif quality_overridden:
        raise ValueError("Passing quality evidence must not claim an override")
    for role in ("teacher", "student"):
        actual_r2 = read_json(root / expected[role + "_validation"] / "metrics.json")["scores"]["rouge2"]
        if quality["scores"][role] != actual_r2:
            raise ValueError("Quality scores differ from actual full validation")
    if abs(quality["scores"]["teacher"] - quality["scores"]["student"] - quality["advantage"]) > 1e-9:
        raise ValueError("Teacher advantage is inconsistent")
    if (markers["smoke"]["identity"] != stage_identity(definition, "smoke", models={
            role: pair["artifacts"][role]["complete_sha256"] for role in ("student", "teacher")})
            or read_json(root / expected["smoke"] / "report.json").get("passed") is not True):
        raise ValueError("Real native GKD save/reload/restart test is missing")
    if full:
        from .evaluation import paired_rows
        paired_rows(root)  # Recheck real prediction coverage/reference hashes and aggregate scores, no ML import.
    approval = pair.get("approval", {})
    if (approval.get("accepted") is not True or approval.get("definition_sha256") != digest(definition)
            or approval.get("quality_complete_sha256") != pair["artifacts"]["quality"]["complete_sha256"]
            or approval.get("smoke_complete_sha256") != pair["artifacts"]["smoke"]["complete_sha256"]):
        raise ValueError("Explicit external-pair approval is missing or stale")
    if quality_overridden:
        if (override is None or approval.get("quality_override_complete_sha256")
                != pair["artifacts"]["quality_override"]["complete_sha256"]):
            raise ValueError("Quality override is not bound to explicit pair approval")
    elif "quality_override_complete_sha256" in approval:
        raise ValueError("Pair approval refers to missing quality-override evidence")
    if pair["runtime"]["environment"]["packages"]["transformers"] != "5.20.2":
        raise ValueError("External pair was not validated with its native runtime")
    if check_runtime and environment() != pair["runtime"]["environment"]:
        raise ValueError("Use the exact validated native external environment, not the old SFT venv")
    result = {"kind": PAIR_KIND, "sft_work": str(root),
            "paths": {"student_checkpoint": str(root / expected["student"]),
                      "teacher_export": str(root / expected["teacher"])},
            "hashes": {"prepared_pair": sha256(root / "prepared_pair.json") if _candidate is None else digest(pair),
                       "student": {"checkpoint": pair["artifacts"]["student"]["complete_sha256"],
                                   "prepared": pair["artifacts"]["prepared"]["complete_sha256"]},
                       "teacher": {"export": pair["artifacts"]["teacher"]["complete_sha256"]}},
            "external_provenance": {"student_mode": definition["student"], "local_sft_updates": 0,
                                    "sources": locked, "approval": approval}}
    if override is not None:
        result["external_provenance"]["quality_override"] = override
    return result


def approve(work):
    root = pair_root(work)
    if (root / "prepared_pair.json").exists():
        return inspect(root)
    definition = request(work)["definition"]
    override = quality_policy.current_override(root, definition)
    names = expected_artifacts(definition, override is not None)
    artifacts = {}
    for name, relative in names.items():
        verify(owned(root, relative))
        artifacts[name] = {"path": relative, "complete_sha256": sha256(owned(root, relative) / "COMPLETE.json")}
    quality = read_json(root / "reports/quality/report.json")
    if quality.get("passed") is not True and override is None:
        raise ValueError("Cannot approve a rejected teacher")
    value = {"kind": PAIR_KIND, "status": "complete", "definition": definition,
             "protocol": PROTOCOL, "local_sft_updates": 0, "artifacts": artifacts,
             "models_lock_sha256": sha256(root / "models.lock.json"),
             "runtime": read_json(Path(work) / "native_runtime.json"),
             "approval": {"accepted": True, "definition_sha256": digest(definition),
                          "quality_complete_sha256": artifacts["quality"]["complete_sha256"],
                          "smoke_complete_sha256": artifacts["smoke"]["complete_sha256"],
                           "unix_time": time.time(), "source_fingerprints": code()}}
    if override is not None:
        value["approval"]["quality_override_complete_sha256"] = artifacts["quality_override"]["complete_sha256"]
    # Validate all backing evidence BEFORE publishing a complete pair.
    inspect(root, _candidate=value)
    atomic_json(root / "prepared_pair.json", value)
    return inspect(root)
