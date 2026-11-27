"""Read-only, best-effort display of quality measurements; no admission decisions."""
from __future__ import annotations

import math
from pathlib import Path

from .common import pair_root, read_json
from sftlib.console import message


def _optional_json(path):
    try:
        value = read_json(path)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        message(f"Optional display unavailable for {path}: {error}")
        return {}
    return value if isinstance(value, dict) else {}


def _text(value):
    if type(value) is float:
        return f"{value:.6f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_text(item) for item in value) + "]"
    return str(value)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def display_quality(work, report):
    """Display available current/cached values without comparing or grading them."""
    root = pair_root(work)
    # Preserve every available report field and its full precision for inspection.
    message(f"Measured pair quality (informational only): {report}")
    report = report if isinstance(report, dict) else {}
    message("Quality measurements — informational only; legacy pass/threshold fields are ignored.")
    metrics = {}
    for role in ("student", "teacher"):
        value = _optional_json(root / "validation" / role / "metrics.json")
        scores = value.get("scores", {})
        scores = scores if isinstance(scores, dict) else {}
        metrics[role] = scores
        available = [f"{key}={_text(scores[key])}" for key in ("rouge1", "rouge2", "rougeL")
                     if scores.get(key) is not None]
        if available:
            message(f"  {role.title()} full validation: " + ", ".join(available))
        if value.get("counts") is not None:
            message(f"  {role.title()} output counts: {value['counts']}")
        if value.get("rates") is not None:
            message(f"  {role.title()} output rates: {value['rates']}")
    student, teacher = (metrics[role].get("rouge2") for role in ("student", "teacher"))
    if _number(student) and _number(teacher):
        message(f"  Full-validation teacher-minus-student R2: {_text(teacher - student)}")
    scores = report.get("scores", {})
    if isinstance(scores, dict):
        for role in ("student", "teacher"):
            if scores.get(role) is not None:
                message(f"  Reported quality-summary {role} R2: {_text(scores[role])}")
    for key, label in (
        ("advantage", "Reported teacher-minus-student R2"),
        ("paired_bootstrap_95ci", "Paired bootstrap 95% CI"),
        ("publisher_teacher_r2", "Publisher teacher R2 (reference)"),
        ("teacher_minus_publisher_r2", "Reported teacher-minus-publisher R2"),
        ("publisher_student_r2", "Publisher public-student R2 (reference)"),
        ("historical_our_student_r2", "Historical our-student R2 (reference)"),
    ):
        if report.get(key) is not None:
            message(f"  {label}: {_text(report[key])}")
    # Older reports do not carry the public student's reference score.
    definition = _optional_json(Path(work) / "run.json").get("definition", {})
    if isinstance(definition, dict) and definition.get("student") == "public" \
            and report.get("publisher_student_r2") is None:
        models = _optional_json(root / "models.lock.json").get("models", {})
        public = models.get("public_student", {}) if isinstance(models, dict) else {}
        reference = public.get("publisher_reported_validation_rouge2") if isinstance(public, dict) else None
        if reference is not None:
            message(f"  Publisher public-student R2 (reference): {_text(reference)}")
    message(f"  Saved quality report: {root / 'reports/quality/report.json'}")
