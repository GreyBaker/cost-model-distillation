"""Audited operator waiver of statistical quality thresholds, never of integrity."""
from __future__ import annotations

import math
from pathlib import Path
import time

from .common import (atomic_json, code, digest, pair_root, publish, read_json, request,
                     sha256, temporary, verify)
from gridlib.policy import PROTOCOL, VALIDATION_N

RELATIVE = "reports/quality_override"
KIND = "external-quality-override-v1"


def failed_gates(report, thresholds):
    """Validate the evidence's internal consistency before considering a waiver."""
    if (not isinstance(thresholds, dict) or set(thresholds) != {"teacher_r2", "advantage", "ci_lower"}
            or type(report.get("n")) is not int or report["n"] != VALIDATION_N
            or "advantage" not in report or report.get("thresholds") != thresholds
            or type(report.get("passed")) is not bool
            or set(report.get("scores", {})) != {"student", "teacher"}
            or not isinstance(report.get("paired_bootstrap_95ci"), list)
            or len(report["paired_bootstrap_95ci"]) != 2):
        raise ValueError("Incomplete/malformed quality evidence cannot be overridden")
    scores, ci = report["scores"], report["paired_bootstrap_95ci"]
    values = [*scores.values(), report["advantage"], *ci, *thresholds.values()]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("Nonfinite quality evidence cannot be overridden")
    if (any(not 0 <= x <= 100 for x in scores.values()) or not -100 <= ci[0] <= ci[1] <= 100
            or abs(scores["teacher"] - scores["student"] - report["advantage"]) > 1e-9):
        raise ValueError("Inconsistent quality scores/interval cannot be overridden")
    failed = [key for key, passed in (
        ("teacher_r2", scores["teacher"] >= thresholds["teacher_r2"]),
        ("advantage", report["advantage"] >= thresholds["advantage"]),
        ("ci_lower", ci[0] > thresholds["ci_lower"]),
    ) if not passed]
    if report["passed"] != (not failed):
        raise ValueError("Quality pass/fail flag disagrees with its measurements")
    return failed


def evidence(root, definition, full=True):
    """Bind a decision to the actual models, prepared data, and full evaluations."""
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
    failed = failed_gates(report, definition["thresholds"])
    for role in ("student", "teacher"):
        metrics = read_json(root / "validation" / role / "metrics.json")
        if report["scores"][role] != metrics["scores"]["rouge2"]:
            raise ValueError("Quality scores differ from full validation; cannot override")
    if full:
        from .evaluation import paired_rows
        paired_rows(root)
    return binding, report, failed


def override_identity(binding):
    return {"definition_sha256": binding["definition_sha256"], "stage": "quality_override",
            "quality_complete_sha256": binding["quality_complete_sha256"]}


def valid_reason(reason):
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 4000:
        raise ValueError("--override-quality requires a nonempty reason of at most 4000 characters")
    return reason.strip()


def current_override(root, definition, full=True):
    directory = Path(root) / RELATIVE
    if not directory.exists():
        return None
    binding, quality, failed = evidence(root, definition, full=full)
    verify(directory, override_identity(binding), full=full)
    value = read_json(directory / "report.json")
    if (not failed or value.get("kind") != KIND or value.get("accepted") is not True
            or value.get("scope") != "quality-thresholds-only"
            or value.get("binding") != binding or value.get("failed_gates") != failed
            or value.get("observed_quality") != quality):
        raise ValueError("Quality override is stale, foreign, or inconsistent")
    valid_reason(value.get("reason"))
    if type(value.get("unix_time")) not in (int, float) or not math.isfinite(value["unix_time"]):
        raise ValueError("Quality override lacks an audit timestamp")
    return value


def record_override(work, reason):
    """Called under the controller lock; never edit the definition or failed report."""
    reason = valid_reason(reason)
    root, definition = pair_root(work), request(work)["definition"]
    directory = root / RELATIVE
    if directory.exists():
        value = current_override(root, definition)
        if value["reason"] != reason:
            raise ValueError("Existing quality override has a different reason; it will not be overwritten")
        return value
    if (root / "prepared_pair.json").exists() or (Path(definition["grid_work"]) / "binding.json").exists():
        raise ValueError("Cannot introduce a quality override after pair publication/grid binding")
    if not (root / "reports/quality/COMPLETE.json").is_file():
        raise ValueError("Run normally first and review the completed quality report before overriding")
    binding, quality, failed = evidence(root, definition)
    if not failed:
        raise ValueError("Quality already passed; an override is unnecessary")
    value = {"kind": KIND, "scope": "quality-thresholds-only", "accepted": True,
             "reason": reason, "unix_time": time.time(), "binding": binding,
             "failed_gates": failed, "observed_quality": quality, "source_fingerprints": code()}
    scratch = temporary(directory)
    atomic_json(scratch / "report.json", value)
    publish(scratch, directory, override_identity(binding))
    return current_override(root, definition)
