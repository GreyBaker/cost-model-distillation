"""Focused informational-quality, integrity and legacy-command regressions; no ML."""
from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import external_teacher
from external_workflow import common, controller, evaluation, pair, quality_policy, reporting
from gridlib import inputs
from quality_override_fixture import RUNTIME, artifact, failed_pair
from sftlib.core import atomic_json, read_json, sha256, tree_manifest


@contextlib.contextmanager
def restored_files(*paths):
    before = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, content in before.items():
            path.write_bytes(content)


def reseal(directory):
    marker = read_json(directory / "COMPLETE.json")
    marker["files"] = tree_manifest(directory, excluded=("COMPLETE.json",))
    atomic_json(directory / "COMPLETE.json", marker)


class PolicyTests(unittest.TestCase):
    def test_quality_numbers_and_old_threshold_results_are_not_admission_gates(self):
        report = {"passed": False, "n": 11332, "scores": {"teacher": 0., "student": 99.},
                  "advantage": -99., "paired_bootstrap_95ci": [-100., -98.],
                  "thresholds": {"teacher_r2": 100., "advantage": 100., "ci_lower": 100.}}
        quality_policy.validate_report(report)
        # Retired pass flags, thresholds and duplicate derived-score consistency
        # comparisons do not control admission.
        quality_policy.validate_report({**report, "passed": True, "advantage": 4.5})
        quality_policy.validate_report({key: value for key, value in report.items()
                                        if key not in ("passed", "thresholds")})
        for change in ({"n": 1}, {"scores": {"teacher": 0.}},
                       {"scores": {"teacher": float("nan"), "student": 99.}},
                       {"advantage": float("inf")}, {"paired_bootstrap_95ci": [float("nan"), 0.]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                quality_policy.validate_report({**report, **change})
        # Display even poor/negative measurements; unavailable optional metrics
        # do not introduce a new quality gate or create any files.
        shown = {**report, "publisher_teacher_r2": 21.37, "teacher_minus_publisher_r2": -21.37,
                 "historical_our_student_r2": 15.45}
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as console:
            work = Path(d) / "absent"
            reporting.display_quality(work, shown)
            reporting.display_quality(work, {"scores": {"student": 0.}})
            self.assertFalse(work.exists())
            atomic_json(work / "pair/validation/student/metrics.json", {})
            (work / "pair/validation/student/metrics.json").write_text("invalid JSON", encoding="utf-8")
            atomic_json(work / "pair/validation/teacher/metrics.json", {"scores": {"rouge2": 20.5}})
            reporting.display_quality(work, shown)
        text = console.getvalue()
        for expected in ("Reported quality-summary teacher R2: 0.000000",
                         "Reported teacher-minus-student R2: -99.000000",
                         "Paired bootstrap 95% CI: [-100.000000, -98.000000]",
                         "Reported teacher-minus-publisher R2: -21.370000",
                         "Historical our-student R2 (reference): 15.450000",
                         "Teacher full validation: rouge2=20.500000"):
            self.assertIn(expected, text)

    def test_old_cli_quality_flags_are_accepted_but_no_longer_configure_gates(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            args = SimpleNamespace(student="public", restored_root=None, reference_python=None,
                settings=common.REPO / "runpod-gkd/settings.json", grid_work=Path(d) / "grid",
                min_teacher_r2=1000., min_advantage=1000., min_ci_lower=1000.)
            self.assertNotIn("thresholds", controller.definition(args))
            argv = ["external_teacher.py", "--pod-disk", d, "--work-dir", str(work),
                    "--student", "public", "--min-teacher-r2", "1000",
                    "--min-advantage", "1000", "--min-ci-lower", "1000", "run", "--override-quality"]
            with patch.object(sys, "argv", argv), patch.object(controller, "run") as run:
                external_teacher.main()
            self.assertTrue(run.called)
            self.assertFalse(work.exists())


class ReportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.disk = Path(self.temp.name)
        self.work = self.disk / "work"
        # Significant (not just roundoff) differences between diagnostic
        # summaries and saved aggregates must not become admission gates.
        self.root, self.definition = failed_pair(
            self.work, metric_offsets={"teacher": -.1, "student": .01})

    def test_cached_failed_reports_resume_without_override_or_score_comparison(self):
        root, work, definition = self.root, self.work, self.definition
        before, run_before = tree_manifest(root), (work / "run.json").read_bytes()
        self.assertEqual(set(controller.reviewed_pair(work)), {"quality"})
        code = self.disk / "mock-code"
        code.mkdir()
        shutil.copyfile(root / "models.lock.json", code / "models.lock.json")
        shutil.copyfile(common.ROOT / "scheme.json", code / "scheme.json")
        grid = Path(definition["grid_work"])
        args = SimpleNamespace(pod_disk=self.disk, work_dir=work, grid_work=grid, student="public",
            restored_root=None, reference_python=None, settings=common.REPO / "runpod-gkd/settings.json",
            accept_pair=False, accept_grid_estimate=True, override_quality=False)
        calls = []
        def worker(runner, stage, python=None, batch=None):
            calls.append(stage)
            return True
        def call(runner, argv, name, **kwargs):
            calls.append(name)
            if name == "grid-bind":
                result = inputs.inspect_pair(root, full=True)
                self.assertEqual(result["external_provenance"]["quality_policy"], "informational-only")
            if name == "grid-pilot":
                atomic_json(grid / "status.json", {"status": "pilot_ready"})
            return True
        current_definition = {key: value for key, value in definition.items() if key != "thresholds"}
        with patch.object(controller, "operational_paths", return_value=(self.disk, work, grid)), \
                patch.object(controller, "ROOT", code), \
                patch.object(controller, "definition", return_value=current_definition), \
                patch.object(controller, "configure_local_storage"), patch.object(controller, "runtime"), \
                patch.object(controller.Runner, "worker", worker), patch.object(controller.Runner, "call", call), \
                patch.object(pair, "environment", return_value=RUNTIME["environment"]), \
                patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()) as console:
            controller.run(args)
            self.assertEqual(calls[-1], "quality")
            self.assertNotIn("smoke", calls)
            self.assertFalse(any(name.startswith("grid-") for name in calls))
            self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_pair_approval")
            self.assertFalse((root / quality_policy.RELATIVE).exists())
            # Even a stale, incomplete old waiver is irrelevant to a new approval.
            atomic_json(root / quality_policy.RELATIVE / "report.json", {"old_incomplete_record": True})
            args.accept_pair, args.accept_grid_estimate = True, False
            controller.run(args)
            self.assertEqual(calls[-3:], ["grid-bind", "grid-pilot", "grid-estimate"])
            self.assertNotIn("grid-run", calls)
            self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_grid_approval")
        text = console.getvalue()
        for expected in ("Student full validation: rouge1=10.010000, rouge2=10.010000, rougeL=10.010000",
                         "Teacher full validation: rouge1=20.900000, rouge2=20.900000, rougeL=20.900000",
                         "Full-validation teacher-minus-student R2: 10.890000",
                         "Reported quality-summary teacher R2: 21.000000",
                         "Reported teacher-minus-student R2: 11.000000",
                         "Paired bootstrap 95% CI: [10.000000, 12.000000]",
                         "Publisher public-student R2 (reference): 16.055376",
                         "informational only"):
            self.assertIn(expected, text)
        manifest = read_json(root / "prepared_pair.json")
        self.assertEqual(manifest["quality_policy"], "informational-only")
        self.assertNotIn("quality_override", manifest["artifacts"])
        self.assertNotIn("smoke", manifest["artifacts"])
        self.assertEqual((work / "run.json").read_bytes(), run_before)
        self.assertEqual(before, {key: value for key, value in tree_manifest(root).items() if key in before})
        self.assertFalse(read_json(root / "reports/quality/report.json")["passed"])

    def test_hashes_full_coverage_and_finite_measurements_remain_required(self):
        root, work = self.root, self.work
        for relative in ("models/teacher_bf16/model.safetensors", "student/prepared/data.json",
                         "validation/student/metrics.json", "reports/quality/report.json"):
            path = root / relative
            with self.subTest(corrupt=relative), restored_files(path), \
                    patch.object(pair, "environment", return_value=RUNTIME["environment"]):
                path.write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    pair.approve(work)
        validation = root / "validation/student"
        quality = root / "reports/quality"
        paths = [validation / name for name in ("predictions.jsonl", "metrics.json", "COMPLETE.json")]
        with restored_files(*paths, quality / "COMPLETE.json"), \
                patch.object(pair, "environment", return_value=RUNTIME["environment"]):
            paths[0].write_text(paths[0].read_text().splitlines()[0] + "\n", encoding="utf-8")
            metrics = read_json(paths[1])
            metrics["predictions_sha256"] = sha256(paths[0])
            atomic_json(paths[1], metrics)
            reseal(validation)
            marker = read_json(quality / "COMPLETE.json")
            marker["identity"]["validation"]["student"] = sha256(validation / "COMPLETE.json")
            atomic_json(quality / "COMPLETE.json", marker)
            with self.assertRaisesRegex(ValueError, "IDs/references"):
                pair.approve(work)
        self.assertFalse((root / "prepared_pair.json").exists())
        # New reports expose measurements, not boolean quality decisions.
        shutil.rmtree(quality)  # Only this test's generated report.
        with patch.object(evaluation, "bootstrap", return_value=[10., 12.]):
            evaluation.quality(work)
        report = read_json(quality / "report.json")
        self.assertEqual(report["policy"], "informational-only")
        self.assertNotIn("passed", report)
        self.assertNotIn("thresholds", report)

    def test_old_published_waivers_are_archived_provenance_not_regraded(self):
        root, work, definition = self.root, self.work, self.definition
        with patch.object(pair, "environment", return_value=RUNTIME["environment"]):
            pair.approve(work)
            binding, quality = quality_policy.evidence(root, definition)
            artifact(root, quality_policy.RELATIVE, quality_policy.override_identity(binding), {
                "report.json": {"kind": quality_policy.KIND, "scope": "quality-thresholds-only",
                    "accepted": True, "binding": binding, "observed_quality": quality,
                    "failed_gates": ["teacher_r2"], "unix_time": 1., "reason": "Archived note"}})
            path = root / "prepared_pair.json"
            manifest = read_json(path)
            manifest.pop("quality_policy")
            override_hash = sha256(root / quality_policy.RELATIVE / "COMPLETE.json")
            manifest["artifacts"]["quality_override"] = {
                "path": quality_policy.RELATIVE, "complete_sha256": override_hash}
            manifest["approval"]["quality_override_complete_sha256"] = override_hash
            atomic_json(path, manifest)
            before = tree_manifest(root)
            result = pair.approve(work)
            self.assertEqual(result["external_provenance"]["quality_override"]["reason"], "Archived note")
            self.assertEqual(tree_manifest(root), before)
            bad = copy.deepcopy(manifest)
            bad["approval"]["quality_override_complete_sha256"] = "0" * 64
            atomic_json(path, bad)
            with self.assertRaisesRegex(ValueError, "override.*approval"):
                pair.inspect(root, full=False)


if __name__ == "__main__":
    unittest.main()
