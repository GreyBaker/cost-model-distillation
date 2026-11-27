"""Focused quality-waiver policy and real-file provenance/controller checks, without ML."""
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
from external_workflow import common, controller, pair, quality_policy
from gridlib import inputs
from quality_override_fixture import RUNTIME, failed_pair
from sftlib.core import atomic_json, read_json, sha256, tree_manifest

def reseal(directory):
    """Only re-seal synthetic scratch files to reach semantic, not just hash, guards."""
    marker = read_json(directory / "COMPLETE.json")
    marker["files"] = tree_manifest(directory, excluded=("COMPLETE.json",))
    atomic_json(directory / "COMPLETE.json", marker)


@contextlib.contextmanager
def restored_files(*paths):
    before = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, content in before.items():
            path.write_bytes(content)


class PolicyTests(unittest.TestCase):
    def test_threshold_failures_are_waivable_but_malformed_evidence_is_not(self):
        thresholds = {"teacher_r2": 25., "advantage": 1., "ci_lower": 0.}
        report = {"passed": False, "n": 11332, "scores": {"teacher": 15., "student": 16.},
                  "advantage": -1., "paired_bootstrap_95ci": [-2., 0.], "thresholds": thresholds}
        self.assertEqual(quality_policy.failed_gates(report, thresholds), list(thresholds))
        passing = {**report, "passed": True, "thresholds": {**thresholds, "teacher_r2": 10.,
                                                        "advantage": -2., "ci_lower": -3.}}
        self.assertEqual(quality_policy.failed_gates(passing, passing["thresholds"]), [])
        changes = [
            ("missing score", {"scores": {"teacher": 15.}}),
            ("incomplete validation", {"n": 1}),
            ("threshold mismatch", {"thresholds": {**thresholds, "teacher_r2": 0.}}),
            ("advantage mismatch", {"advantage": 500.}),
            ("false pass flag", {"passed": True}),
            ("reversed interval", {"paired_bootstrap_95ci": [5., 4.]}),
            ("nonfinite score", {"scores": {"teacher": float("nan"), "student": 16.}}),
            ("nonfinite advantage", {"advantage": float("inf")}),
            ("nonfinite interval", {"paired_bootstrap_95ci": [float("-inf"), 0.]}),
        ]
        for name, change in changes:
            with self.subTest(name=name), self.assertRaises(ValueError):
                quality_policy.failed_gates({**report, **change}, thresholds)

    def test_boolean_flag_requires_completed_failure_and_unpublished_pair(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "work"
            root = work / "pair"
            definition = {"grid_work": str(Path(d) / "grid")}
            atomic_json(work / "run.json", {"definition": definition})
            with self.assertRaisesRegex(ValueError, "Run normally first"):
                quality_policy.record_override(work)
            for path in (root / "prepared_pair.json", Path(definition["grid_work"]) / "binding.json"):
                atomic_json(path, {})
                with self.subTest(publication=path.name), self.assertRaisesRegex(ValueError, "after pair publication"):
                    quality_policy.record_override(work)
                path.unlink()
            atomic_json(root / "reports/quality/COMPLETE.json", {})
            with patch.object(quality_policy, "evidence", return_value=({}, {"passed": True}, [])), \
                    self.assertRaisesRegex(ValueError, "already passed"):
                quality_policy.record_override(work)
            self.assertFalse((root / quality_policy.RELATIVE).exists())
            for enabled in (False, True):
                argv = ["external_teacher.py", "--pod-disk", d, "--work-dir", str(work),
                        "--student", "public", "run"]
                if enabled:
                    argv.append("--override-quality")
                with patch.object(sys, "argv", argv), patch.object(controller, "run") as run:
                    external_teacher.main()
                self.assertIs(run.call_args.args[0].override_quality, enabled)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.disk = Path(self.temp.name)
        self.work = self.disk / "work"
        self.root, self.definition = failed_pair(self.work)

    def test_audited_override_resumes_unchanged_and_requires_separate_approvals(self):
        root, work, definition = self.root, self.work, self.definition
        before, run_before = tree_manifest(root), (work / "run.json").read_bytes()
        # Pair/grid approval flags must not substitute for the explicit quality decision.
        with self.assertRaisesRegex(ValueError, "valid quality override"):
            controller.reviewed_pair(work)
        with self.assertRaisesRegex(ValueError, "rejected teacher"):
            pair.approve(work)
        code = self.disk / "mock-code"
        code.mkdir()
        shutil.copyfile(root / "models.lock.json", code / "models.lock.json")
        shutil.copyfile(common.ROOT / "scheme.json", code / "scheme.json")
        grid = Path(definition["grid_work"])
        args = SimpleNamespace(pod_disk=self.disk, work_dir=work, grid_work=grid, student="public",
            restored_root=None, reference_python=None, settings=common.REPO / "runpod-gkd/settings.json",
             accept_pair=False, accept_grid_estimate=True, override_quality=True)
        calls = []

        def worker(runner, stage, python=None, batch=None):
            # No model/data work, but the waiver must already exist before any stage.
            self.assertTrue((root / quality_policy.RELATIVE / "COMPLETE.json").is_file())
            calls.append(stage)
            return True

        def call(runner, argv, name, **kwargs):
            calls.append(name)
            if name == "grid-bind":
                result = inputs.inspect_pair(root, full=True)
                self.assertTrue(result["external_provenance"]["quality_override"]["accepted"])
                self.assertNotIn("reason", result["external_provenance"]["quality_override"])
                self.assertFalse(result["external_provenance"]["quality_override"]["observed_quality"]["passed"])
            if name == "grid-pilot":
                atomic_json(grid / "status.json", {"status": "pilot_ready"})
            return True

        with patch.object(controller, "operational_paths", return_value=(self.disk, work, grid)), \
                patch.object(controller, "ROOT", code), patch.object(controller, "definition", return_value=definition), \
                patch.object(controller, "configure_local_storage"), patch.object(controller, "runtime"), \
                patch.object(controller.Runner, "worker", worker), patch.object(controller.Runner, "call", call), \
                patch.object(pair, "environment", return_value=RUNTIME["environment"]), \
                patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()) as console:
            controller.run(args)
            self.assertEqual(calls[-1], "smoke")
            self.assertFalse(any(name.startswith("grid-") for name in calls))
            self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_pair_approval")
            value = quality_policy.current_override(root, definition)
            self.assertEqual(value["failed_gates"], ["teacher_r2"])
            self.assertNotIn("reason", value)
            self.assertEqual(value["observed_quality"], read_json(root / "reports/quality/report.json"))
            self.assertEqual(value["binding"], {
                "definition_sha256": common.digest(definition),
                "prepared_complete_sha256": sha256(root / "student/prepared/COMPLETE.json"),
                "models_complete_sha256": {
                    role: sha256(root / "models" / directory / "COMPLETE.json")
                    for role, directory in (("student", "student_fp32"), ("teacher", "teacher_bf16"))},
                "validation_complete_sha256": {
                    role: sha256(root / "validation" / role / "COMPLETE.json") for role in ("student", "teacher")},
                "quality_complete_sha256": sha256(root / "reports/quality/COMPLETE.json")})
            override_before = tree_manifest(root / quality_policy.RELATIVE)
            self.assertEqual(quality_policy.record_override(work), value)
            # Older records with free text remain valid and are not rewritten.
            report_path = root / quality_policy.RELATIVE / "report.json"
            marker_path = report_path.parent / "COMPLETE.json"
            legacy = {**value, "reason": "An existing optional note"}
            with restored_files(report_path, marker_path):
                atomic_json(report_path, legacy)
                reseal(report_path.parent)
                self.assertEqual(quality_policy.record_override(work), legacy)
            args.override_quality = False
            args.accept_pair, args.accept_grid_estimate = True, False
            controller.run(args)
            self.assertEqual(calls[-3:], ["grid-bind", "grid-pilot", "grid-estimate"])
            self.assertNotIn("grid-run", calls)
            self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_grid_approval")
            self.assertEqual(tree_manifest(root / quality_policy.RELATIVE), override_before)
        self.assertNotIn("QUALITY FAILURE OVERRIDDEN", console.getvalue())
        manifest_path = root / "prepared_pair.json"
        manifest = read_json(manifest_path)
        self.assertEqual(manifest["definition"], definition)
        self.assertEqual(manifest["approval"]["quality_override_complete_sha256"],
                         sha256(root / quality_policy.RELATIVE / "COMPLETE.json"))
        for unbound in ("missing", "wrong"):
            changed = copy.deepcopy(manifest)
            if unbound == "missing":
                del changed["approval"]["quality_override_complete_sha256"]
            else:
                changed["approval"]["quality_override_complete_sha256"] = "0" * 64
            with self.subTest(approval=unbound), restored_files(manifest_path):
                atomic_json(manifest_path, changed)
                with self.assertRaisesRegex(ValueError, "override.*approval"):
                    pair.inspect(root, full=False, check_runtime=False)
        self.assertEqual((work / "run.json").read_bytes(), run_before)
        self.assertEqual(before, {key: item for key, item in tree_manifest(root).items() if key in before})
        self.assertFalse(read_json(root / "reports/quality/report.json")["passed"])

    def test_corrupt_incomplete_or_inconsistent_evidence_cannot_be_waived(self):
        root, work = self.root, self.work
        for relative in ("models/teacher_bf16/model.safetensors", "student/prepared/data.json",
                         "validation/student/metrics.json", "reports/quality/report.json"):
            path = root / relative
            with self.subTest(corrupt=relative), restored_files(path):
                path.write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    quality_policy.record_override(work)
        marker = root / "reports/quality/COMPLETE.json"
        with restored_files(marker):
            marker.unlink()
            with self.assertRaisesRegex(ValueError, "Run normally first"):
                quality_policy.record_override(work)
        quality = root / "reports/quality"
        report_path = quality / "report.json"
        with restored_files(report_path, quality / "COMPLETE.json"):
            report = read_json(report_path)
            report["scores"]["teacher"], report["advantage"] = 20., 10.
            atomic_json(report_path, report)
            reseal(quality)
            with self.assertRaisesRegex(ValueError, "scores differ from full validation"):
                quality_policy.record_override(work)
        # Internally re-seal the short evaluation and rebind the quality marker so
        # rejection must come from real coverage validation, not a stale hash.
        validation = root / "validation/student"
        paths = [validation / name for name in ("predictions.jsonl", "metrics.json", "COMPLETE.json")]
        with restored_files(*paths, quality / "COMPLETE.json"):
            paths[0].write_text(paths[0].read_text().splitlines()[0] + "\n", encoding="utf-8")
            metrics = read_json(paths[1])
            metrics["predictions_sha256"] = sha256(paths[0])
            atomic_json(paths[1], metrics)
            reseal(validation)
            marker = read_json(quality / "COMPLETE.json")
            marker["identity"]["validation"]["student"] = sha256(validation / "COMPLETE.json")
            atomic_json(quality / "COMPLETE.json", marker)
            with self.assertRaisesRegex(ValueError, "IDs/references"):
                quality_policy.record_override(work)
        self.assertFalse((root / quality_policy.RELATIVE).exists())

    def test_saved_override_binding_integrity_and_smoke_guards_remain_mandatory(self):
        root, work, definition = self.root, self.work, self.definition
        quality_policy.record_override(work)
        directory = root / quality_policy.RELATIVE
        report_path, marker_path = directory / "report.json", directory / "COMPLETE.json"
        original = read_json(report_path)
        for component in original["binding"]:
            changed = copy.deepcopy(original)
            binding = changed["binding"]
            if isinstance(binding[component], dict):
                binding[component]["teacher"] = "0" * 64
            else:
                binding[component] = "0" * 64
            with self.subTest(stale=component), restored_files(report_path, marker_path):
                atomic_json(report_path, changed)
                reseal(directory)
                with self.assertRaisesRegex(ValueError, "stale"):
                    quality_policy.current_override(root, definition)
        with restored_files(report_path):
            atomic_json(report_path, {**original, "accepted": False})
            with self.assertRaises(ValueError):
                quality_policy.current_override(root, definition)
        smoke = root / "reports/smoke"
        report_path, marker_path = smoke / "report.json", smoke / "COMPLETE.json"
        for state in ("missing", "failed"):
            with self.subTest(smoke=state), restored_files(report_path, marker_path):
                if state == "missing":
                    marker_path.unlink()
                else:
                    atomic_json(report_path, {"passed": False})
                    reseal(smoke)
                with self.assertRaises(ValueError):
                    controller.reviewed_pair(work)
                with self.assertRaises((ValueError, OSError)):
                    pair.approve(work)
                self.assertFalse((root / "prepared_pair.json").exists())


if __name__ == "__main__":
    unittest.main()
