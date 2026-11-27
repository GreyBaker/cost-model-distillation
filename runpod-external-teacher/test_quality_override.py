"""Quality waivers are audited decisions, not edits to failed scientific evidence."""
from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_external import RUNTIME, fixture
from external_workflow import common, controller, pair, quality_policy
from gridlib import inputs
from sftlib.core import atomic_json, read_json, sha256, tree_manifest

REASON = "Reviewed the full results; accepting this teacher for the planned controlled grid."


def failed_pair(work, **kwargs):
    root, manifest = fixture(work, thresholds={"teacher_r2": 25., "advantage": 1., "ci_lower": 0.}, **kwargs)
    (root / "prepared_pair.json").unlink()  # Only the manifest created by this test fixture.
    return root, manifest["definition"]


class OverrideTests(unittest.TestCase):
    def test_records_reason_without_changing_failed_report_definition_or_evaluations(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            root, definition = failed_pair(work)
            before = tree_manifest(root)
            run_before = (work / "run.json").read_bytes()
            value = quality_policy.record_override(work, REASON)
            self.assertEqual(value["failed_gates"], ["teacher_r2"])
            self.assertEqual(value["reason"], REASON)
            self.assertFalse(value["observed_quality"]["passed"])
            self.assertFalse(read_json(root / "reports/quality/report.json")["passed"])
            self.assertEqual((work / "run.json").read_bytes(), run_before)
            self.assertEqual(before, {key: item for key, item in tree_manifest(root).items() if key in before})
            self.assertEqual(value["binding"]["definition_sha256"], common.digest(definition))
            sealed = sha256(root / quality_policy.RELATIVE / "COMPLETE.json")
            quality_policy.record_override(work, REASON)
            self.assertEqual(sha256(root / quality_policy.RELATIVE / "COMPLETE.json"), sealed)
            with self.assertRaisesRegex(ValueError, "different reason"):
                quality_policy.record_override(work, "An alternative reason must not overwrite the original.")

    def test_override_is_included_in_pair_approval_and_grid_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            root, _ = failed_pair(work)
            quality_policy.record_override(work, REASON)
            reviewed = controller.reviewed_pair(work)
            self.assertIn("quality_override", reviewed)
            with patch.object(pair, "environment", return_value=RUNTIME["environment"]):
                pair.approve(work)
                result = inputs.inspect_pair(root, full=True)
            manifest = read_json(root / "prepared_pair.json")
            self.assertEqual(manifest["approval"]["quality_override_complete_sha256"], reviewed["quality_override"])
            self.assertEqual(result["external_provenance"]["quality_override"]["reason"], REASON)
            self.assertFalse(result["external_provenance"]["quality_override"]["observed_quality"]["passed"])
            self.assertEqual(manifest["definition"]["thresholds"]["teacher_r2"], 25.)

    def test_cannot_override_before_complete_evaluation_or_after_publication(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "empty"
            atomic_json(work / "run.json", {"definition": {"grid_work": str(Path(d) / "grid")}})
            with self.assertRaisesRegex(ValueError, "Run normally first"):
                quality_policy.record_override(work, REASON)
            self.assertFalse((work / "pair" / quality_policy.RELATIVE).exists())
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            fixture(work)  # Published passing fixture.
            with self.assertRaisesRegex(ValueError, "after pair publication"):
                quality_policy.record_override(work, REASON)

    def test_empty_reason_passing_quality_and_missing_prediction_coverage_rejected(self):
        for kind in ("reason", "passed", "coverage"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                work = Path(d) / "work"
                if kind == "passed":
                    root, _ = fixture(work)
                    (root / "prepared_pair.json").unlink()
                else:
                    root, _ = failed_pair(work, full_rows=kind != "coverage")
                with self.assertRaises(ValueError):
                    quality_policy.record_override(work, "  " if kind == "reason" else REASON)
                self.assertFalse((root / quality_policy.RELATIVE).exists())

    def test_corruption_in_weights_and_report_is_not_a_quality_override(self):
        for relative in ("models/teacher_bf16/model.safetensors", "reports/quality/report.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as d:
                work = Path(d) / "work"
                root, _ = failed_pair(work)
                (root / relative).write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    quality_policy.record_override(work, REASON)
                self.assertFalse((root / quality_policy.RELATIVE).exists())

    def test_waiver_does_not_waive_a_missing_or_failed_smoke_test(self):
        from test_external import artifact
        for kind in ("missing", "failed"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                work = Path(d) / "work"
                root, definition = failed_pair(work)
                quality_policy.record_override(work, REASON)
                shutil.rmtree(root / "reports/smoke")  # Only this test's scratch artifact.
                if kind == "failed":
                    artifact(root, "reports/smoke", pair.stage_identity(definition, "smoke", models={
                        role: sha256(root / "models" / ("student_fp32" if role == "student" else "teacher_bf16")
                                     / "COMPLETE.json") for role in ("student", "teacher")}),
                        {"report.json": {"passed": False}})
                with self.assertRaises(ValueError):
                    controller.reviewed_pair(work)
                with self.assertRaises((ValueError, OSError)):
                    pair.approve(work)
                self.assertFalse((root / "prepared_pair.json").exists())

    def test_stale_or_tampered_override_and_unbound_approval_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            root, definition = failed_pair(work)
            quality_policy.record_override(work, REASON)
            with patch.object(pair, "environment", return_value=RUNTIME["environment"]):
                pair.approve(work)
            path = root / "prepared_pair.json"
            original = read_json(path)
            for changed in ({**original["approval"], "quality_override_complete_sha256": "0" * 64},
                            {key: value for key, value in original["approval"].items()
                             if key != "quality_override_complete_sha256"}):
                atomic_json(path, {**original, "approval": changed})
                with self.assertRaisesRegex(ValueError, "override.*approval"):
                    pair.inspect(root, full=False, check_runtime=False)
            report_path = root / quality_policy.RELATIVE / "report.json"
            report = read_json(report_path)
            report["binding"]["quality_complete_sha256"] = "0" * 64
            atomic_json(report_path, report)
            # Re-seal only our synthetic fixture to test the binding check,
            # independently of the simpler corrupted-file hash check.
            marker_path = report_path.parent / "COMPLETE.json"
            marker = read_json(marker_path)
            marker["files"] = tree_manifest(report_path.parent, excluded=("COMPLETE.json",))
            atomic_json(marker_path, marker)
            with self.assertRaisesRegex(ValueError, "stale"):
                quality_policy.current_override(root, definition)

    def test_nonfinite_or_inconsistent_scores_and_flags_cannot_be_waived(self):
        report = {"passed": False, "n": 11332, "scores": {"teacher": 20.5, "student": 16.},
                  "advantage": 4.5, "paired_bootstrap_95ci": [4., 5.],
                  "thresholds": {"teacher_r2": 25., "advantage": 1., "ci_lower": 0.}}
        self.assertEqual(quality_policy.failed_gates(report, report["thresholds"]), ["teacher_r2"])
        for kind in ("nonfinite", "advantage", "flag", "interval"):
            changed = copy.deepcopy(report)
            if kind == "nonfinite":
                changed["scores"]["teacher"] = float("nan")
            elif kind == "advantage":
                changed["advantage"] = 500.
            elif kind == "flag":
                changed["passed"] = True
            else:
                changed["paired_bootstrap_95ci"] = [5., 4.]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                quality_policy.failed_gates(changed, report["thresholds"])

    def test_all_three_quality_thresholds_are_discretionary_not_integrity_checks(self):
        thresholds = {"teacher_r2": 25., "advantage": 1., "ci_lower": 0.}
        report = {"passed": False, "n": 11332, "scores": {"teacher": 15., "student": 16.},
                  "advantage": -1., "paired_bootstrap_95ci": [-2., 0.], "thresholds": thresholds}
        self.assertEqual(quality_policy.failed_gates(report, thresholds), ["teacher_r2", "advantage", "ci_lower"])


class ControllerOverrideTests(unittest.TestCase):
    def test_override_runs_smoke_then_requires_pair_and_grid_estimate_approvals(self):
        with tempfile.TemporaryDirectory() as d:
            disk = Path(d)
            work, grid = disk / "work", disk / "grid"
            root, definition = failed_pair(work)
            code = disk / "mock-code"
            code.mkdir()
            shutil.copyfile(root / "models.lock.json", code / "models.lock.json")
            shutil.copyfile(common.ROOT / "scheme.json", code / "scheme.json")
            args = SimpleNamespace(pod_disk=disk, work_dir=work, grid_work=grid, student="public",
                restored_root=None, reference_python=None, settings=common.REPO / "runpod-gkd/settings.json",
                accept_pair=False, accept_grid_estimate=False, override_quality=REASON)
            calls = []
            def worker(self, stage, python=None, batch=None):
                calls.append(stage)  # All GPU/data stages mocked; existing full evidence stays byte-identical.
                return True
            def call(self, argv, name, **kwargs):
                calls.append(name)
                if name == "grid-bind":
                    inputs.inspect_pair(root, full=True)
                if name == "grid-pilot":
                    atomic_json(grid / "status.json", {"status": "pilot_ready"})
                return True
            before = tree_manifest(root / "validation")
            with patch.object(controller, "operational_paths", return_value=(disk, work, grid)), \
                    patch.object(controller, "ROOT", code), patch.object(controller, "definition", return_value=definition), \
                    patch.object(controller, "configure_local_storage"), patch.object(controller, "runtime"), \
                    patch.object(controller.Runner, "worker", worker), patch.object(controller.Runner, "call", call), \
                    patch.object(pair, "environment", return_value=RUNTIME["environment"]), \
                    patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
                controller.run(args)
                self.assertEqual(calls[-1], "smoke")
                self.assertNotIn("grid-bind", calls)
                self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_pair_approval")
                args.override_quality = None  # Persisted, report-bound decision is reused on resume.
                args.accept_pair = True
                controller.run(args)
                self.assertEqual(calls[-3:], ["grid-bind", "grid-pilot", "grid-estimate"])
                self.assertNotIn("grid-run", calls)
                self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_grid_approval")
            self.assertEqual(tree_manifest(root / "validation"), before)
            self.assertFalse(read_json(root / "reports/quality/report.json")["passed"])


if __name__ == "__main__":
    unittest.main()
