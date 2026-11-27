"""Model-free regressions for once-only throughput selection and decoding options."""
from __future__ import annotations

import contextlib
import importlib.abc
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridlib import backend, evalpilot, policy, recovery, schedule
from gridlib.shared import atomic_json, read_json, sha256, tree_manifest
from sftlib import modeling


def fixture(root, *, cap=256):
    atomic_json(root / "binding.json", {"inputs": "immutable", "training_design": schedule.DESIGN})
    atomic_json(root / "pilot.json", {"training": "immutable"})
    selected = {"microbatch": 4, "runtime": {"gpu": "mock", "total_memory": 1000}}
    bound = {"settings": {**policy.settings(), "evaluation_batch_cap": cap},
             "training_design": schedule.DESIGN,
             "code": {"generation": "before"},
             "inputs": {"hashes": {"student": {"checkpoint": "immutable-student"}}}}
    return bound, selected


def runner(root, bound, selected, calls, rates=None, *, stop_after=None, limit=160):
    rates = rates or {}
    def call(operation, owner, *extra):
        assert (operation, owner) == ("evaluation_pilot", "shared")
        batch = int(extra[extra.index("--microbatch") + 1])
        calls.append(batch)
        if stop_after is not None and len(calls) > stop_after:
            return False
        result = {"request": evalpilot.request(root, bound, selected),
                  "batch_size": batch, "status": "oom" if batch > limit else "pass"}
        if batch <= limit:
            rate = rates.get(batch, 10.)
            # Huge first warmup must not penalize an otherwise fast candidate.
            result.update(runtime=selected["runtime"], peak_reserved_bytes=100,
                          generation_seconds=[100000., batch / rate, batch / rate],
                          warmed_summaries_per_second=rate)
        atomic_json(Path(extra[extra.index("--output") + 1]), result)
        return True
    return SimpleNamespace(call=call)


class ThroughputTests(unittest.TestCase):
    def ensure(self, root, bound, selected, calls, rates=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return evalpilot.ensure(root, bound, selected, runner(root, bound, selected, calls, rates, **kwargs))

    def test_new_pilot_selects_faster_smaller_batch_not_largest_fitting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root)
            calls = []
            self.assertTrue(self.ensure(root, bound, selected, calls, {32: 90., 64: 100., 144: 80.}))
            value = evalpilot.saved(root, bound, selected)
            self.assertEqual(value["request"]["version"], 2)
            self.assertEqual(value["max_passing_batch"], 160)
            self.assertEqual(value["safety_ceiling_batch"], 144)
            self.assertEqual(value["batch_size"], 64)
            self.assertEqual(value["fastest_measured_batch"], 64)
            self.assertEqual(value["warmed_summaries_per_second"], 100.)
            self.assertEqual(len(calls), len(set(calls)))
            for item in [None, *schedule.milestones()]:
                self.assertEqual(recovery.evaluation_microbatch(root, bound, item, selected), 64)
            self.assertEqual(recovery.training_microbatch(root, selected), 4)

    def test_near_best_plateau_prefers_smaller_memory_footprint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root)
            self.ensure(root, bound, selected, [], {32: 99., 64: 100.})
            value = evalpilot.saved(root, bound, selected)
            self.assertEqual(value["fastest_measured_batch"], 64)
            self.assertEqual(value["batch_size"], 32)
            self.assertAlmostEqual(value["warmed_summaries_per_second"], 99.)

    def test_faster_batch_outside_safety_ceiling_is_ineligible(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root)
            self.ensure(root, bound, selected, [], {64: 100., 160: 10000.})
            self.assertEqual(evalpilot.saved(root, bound, selected)["batch_size"], 64)

    def test_eligible_binary_fit_measurement_also_competes_on_speed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root, cap=128)
            self.ensure(root, bound, selected, [], {64: 85., 96: 100., 99: 90.}, limit=110)
            value = evalpilot.saved(root, bound, selected)
            self.assertEqual(value["safety_ceiling_batch"], 99)
            self.assertNotIn(96, evalpilot.speed_candidates(99))
            self.assertEqual(value["batch_size"], 96)

    def test_completed_new_pilot_reused_after_code_change_without_new_probes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root)
            calls = []
            self.ensure(root, bound, selected, calls, {64: 100.})
            before = sha256(root / "evaluation_pilot.json")
            bound["code"] = {"generation": "after"}
            previous = list(calls)
            self.ensure(root, bound, selected, calls, {144: 10000.})
            self.assertEqual(calls, previous)
            self.assertEqual(sha256(root / "evaluation_pilot.json"), before)

    def test_completed_and_partial_capacity_only_pilots_are_rejected_without_recalibration(self):
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                bound, selected = fixture(root)
                request = {**evalpilot.request(root, bound, selected), "version": 1}
                request.pop("selection_policy")
                request.pop("throughput_near_best_fraction")
                if partial:
                    atomic_json(root / "evaluation_pilot_request.json", request)
                else:
                    atomic_json(root / "evaluation_pilot.json", {"request": request})
                before, calls = tree_manifest(root), []
                with self.assertRaises(ValueError):
                    self.ensure(root, bound, selected, calls)
                self.assertFalse(calls)
                self.assertEqual(tree_manifest(root), before)

    def test_partial_throughput_sweep_resumes_without_repeating_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root, cap=64)
            first, second = [], []
            # Seven exponential probes then ceiling57, then new quarter14.
            self.assertFalse(self.ensure(root, bound, selected, first, {28: 100.}, stop_after=9))
            self.assertTrue(self.ensure(root, bound, selected, second, {28: 100.}))
            self.assertTrue(set(first[:9]).isdisjoint(second))
            self.assertEqual(evalpilot.saved(root, bound, selected)["batch_size"], 28)

    def test_missing_speed_probe_or_fabricated_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root, cap=64)
            self.ensure(root, bound, selected, [], {28: 100.})
            path = root / "evaluation_pilot.json"
            original = read_json(path)
            missing = {**original, "result_files": [
                entry for entry in original["result_files"]
                if read_json(Path(entry["path"]))["batch_size"] != 42]}
            atomic_json(path, missing)
            with self.assertRaisesRegex(ValueError, "required probe batch 42"):
                evalpilot.saved(root, bound, selected)
            for changes in ({"batch_size": 57}, {"safety_ceiling_batch": 58},
                            {"fastest_measured_batch": 57}, {"warmed_summaries_per_second": 200.}):
                atomic_json(path, {**original, **changes})
                with self.assertRaisesRegex(ValueError, "deterministic search"):
                    evalpilot.saved(root, bound, selected)

    def test_timing_validation_cannot_trust_a_fabricated_throughput_scalar(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound, selected = fixture(root)
            expected = evalpilot.request(root, bound, selected)
            base = {"request": expected, "batch_size": 4, "status": "pass",
                    "runtime": selected["runtime"], "peak_reserved_bytes": 100,
                    "generation_seconds": [100., 1., 1.], "warmed_summaries_per_second": 1000.}
            with self.assertRaisesRegex(ValueError, "throughput"):
                evalpilot.verify_trial(base, expected, 4)
            for bad in (float("nan"), 0., -1., True):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    evalpilot.verify_trial({**base, "generation_seconds": [100., bad, 1.]}, expected, 4)

    def test_probe_schedule_is_bounded_including_non_power_of_two_batches(self):
        for cap in (1, 8, 256, 11332):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                bound, selected = fixture(root, cap=cap)
                calls = []
                self.ensure(root, bound, selected, calls, limit=cap)
                self.assertLessEqual(len(calls), 3 * math.ceil(math.log2(cap + 1)) + 5)
                self.assertEqual(evalpilot.saved(root, bound, selected)["max_passing_batch"], cap)


class GenerationTests(unittest.TestCase):
    def test_no_redundant_min_length_in_greedy_sampling_or_stress_generation(self):
        class IDs:
            def __getitem__(self, index):
                return self
            def eq(self, value):
                return self
            def all(self):
                return True
        class Model:
            config = SimpleNamespace(decoder_start_token_id=0)
            training = True
            def eval(self):
                self.training = False
            def train(self, mode):
                self.training = mode
            def generate(self, **kwargs):
                return IDs()
        configs = []
        def generation_config(**kwargs):
            self.assertNotIn("min_length", kwargs)
            configs.append(kwargs)
            return SimpleNamespace(**kwargs)
        torch = SimpleNamespace(no_grad=contextlib.nullcontext, bfloat16="bf16",
                                autocast=lambda *a, **k: contextlib.nullcontext())
        tok, model = SimpleNamespace(eos_token_id=1, pad_token_id=0), Model()
        with patch.dict("sys.modules", {"torch": torch,
                "transformers": SimpleNamespace(GenerationConfig=generation_config)}):
            modeling.generate(model, tok, {}, 64)
            backend.generate(model, tok, {}, sample=False)
            backend.generate(model, tok, {}, sample=True)
            backend.generate(model, tok, {}, sample=False, force_cap=True)
        self.assertEqual([value["min_new_tokens"] for value in configs], [0, 0, 0, 63])
        self.assertEqual(configs[2]["temperature"], 1.)
        self.assertEqual(configs[2]["top_k"], 0)
        self.assertTrue(model.training)


if __name__ == "__main__":
    import os
    preferred = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp/opencode"
    if os.name == "nt" and preferred.is_dir():
        tempfile.tempdir = str(preferred)
    class NoML(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "transformers", "datasets", "numpy", "accelerate",
                                         "pyarrow", "safetensors", "tokenizers", "rouge_score", "sentencepiece"}:
                raise AssertionError("Real ML import blocked: " + fullname)
    sys.meta_path.insert(0, NoML())
    unittest.main()
