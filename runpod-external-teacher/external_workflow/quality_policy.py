"""Informational quality reports and read-only compatibility for archived waivers."""
from __future__ import annotations

import math
from pathlib import Path
from .common import digest, read_json, sha256, verify
from gridlib.policy import PROTOCOL, VALIDATION_N

RELATIVE = "reports/quality_override"
KIND = "external-quality-override-v1"
def validate_report(report):
    """Check completeness/finite measurements, not performance or derived equality."""
    if (type(report.get("n")) is not int or report["n"] != VALIDATION_N
            or "advantage" not in report or set(report.get("scores", {})) != {"student", "teacher"}
            or not isinstance(report.get("paired_bootstrap_95ci"), list)
            or len(report["paired_bootstrap_95ci"]) != 2):
        raise ValueError("Incomplete/malformed quality report")
    scores, ci = report["scores"], report["paired_bootstrap_95ci"]
    values = [*scores.values(), report["advantage"], *ci]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite quality report")


def evidence(root, definition, full=True):
    """Bind archived provenance to actual models, data and complete evaluations."""
    root = Path(root)
    definition_sha = digest(definition)
    prepared = root / "student/prepared"
    verify(prepared, {"definition_sha256": definition_sha, "stage": "prepared"}, full=full)
    binding = {"definition_sha256": definition_sha,
               "prepared_complete_sha256": sha256(prepared / "COMPLETE.json"),
               "models_complete_sha256": {}, "validation_complete_sha256": {}}
    for role in ("student", "teacher"):
        model = root / "models" / ("student_fp32" if role == "student" else "teacher_bf16")
        verify(model, {"definition_sha256": definition_sha, "stage": "model", "role": role}, full=full)
        model_hash = sha256(model / "COMPLETE.json")
        validation = root / "validation" / role
        verify(validation, {"definition_sha256": definition_sha, "stage": "validation", "role": role,
            "model_complete_sha256": model_hash,
            "data_complete_sha256": binding["prepared_complete_sha256"], "protocol": PROTOCOL}, full=full)
        binding["models_complete_sha256"][role] = model_hash
        binding["validation_complete_sha256"][role] = sha256(validation / "COMPLETE.json")
    quality = root / "reports/quality"
    verify(quality, {"definition_sha256": definition_sha, "stage": "quality",
                     "validation": binding["validation_complete_sha256"]}, full=full)
    binding["quality_complete_sha256"] = sha256(quality / "COMPLETE.json")
    report = read_json(quality / "report.json")
    validate_report(report)
    if full:
        from .evaluation import paired_rows
        paired_rows(root)
    return binding, report


def override_identity(binding):
    return {"definition_sha256": binding["definition_sha256"], "stage": "quality_override",
            "quality_complete_sha256": binding["quality_complete_sha256"]}


def current_override(root, definition, full=True):
    """Read an already-published waiver as provenance only, never as a quality gate."""
    directory = Path(root) / RELATIVE
    if not directory.exists():
        return None
    binding, quality = evidence(root, definition, full=full)
    verify(directory, override_identity(binding), full=full)
    value = read_json(directory / "report.json")
    if (value.get("kind") != KIND or value.get("accepted") is not True
            or value.get("scope") != "quality-thresholds-only"
            or value.get("binding") != binding
            or value.get("observed_quality") != quality):
        raise ValueError("Quality override is stale, foreign, or inconsistent")
    if type(value.get("unix_time")) not in (int, float) or not math.isfinite(value["unix_time"]):
        raise ValueError("Quality override lacks an audit timestamp")
    return value
