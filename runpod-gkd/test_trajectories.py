"""Focused continuous-trajectory regressions; fake tensors/optimizer, no ML or GPU."""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridlib import (backend, controller, engine, evaluation, inputs, milestones,
                     policy, recovery, reporting, schedule, trajectory_reporting)
from gridlib.shared import atomic_json, checkpoints, read_json, sha256, tree_manifest
from sftlib import training
import distill_grid


class Model:
    def __init__(self, count=0):
        self.count = count

    def parameters(self):
        return [SimpleNamespace(numel=lambda: 1, element_size=lambda: 4)]

    def save_pretrained(self, path, **kwargs):
        (path / "model.safetensors").write_text(str(self.count))


class Optimizer:
    def __init__(self):
        self.count = 0
        self.param_groups = [{"lr": 0.}]

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, value):
        self.count = value["count"]


@contextlib.contextmanager
def fake_training(root):
    bound = {"settings": policy.settings(), "protocol": policy.PROTOCOL,
             "training_design": schedule.DESIGN,
             "inputs": {"paths": {"student_checkpoint": "original-fp32"}, "sft_work": "pair"}}
    selected = {"microbatch": 32, "runtime": {"mock": True}}
    atomic_json(root / "binding.json", bound)
    atomic_json(root / "pilot.json", {"mock": True})
    generator = random.Random()
    calls, loads = [], []
    tokenizer = SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer.json").write_text("{}"))
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: None, reset_peak_memory_stats=lambda device: None),
        save=lambda state, path: path.write_bytes(pickle.dumps(state)),
        load=lambda path, **kwargs: pickle.loads(path.read_bytes()))

    def load(bound, settings, checkpoint=None):
        loads.append(checkpoint)
        count = int((checkpoint / "model.safetensors").read_text()) if checkpoint else 0
        return "mock", selected["runtime"], tokenizer, Model(count), None, {"train": list(range(128))}

    def update(student, teacher, tokenizer, opt, batches, n, device):
        student.count += 1
        opt.count += 1
        calls.append({"model": student.count, "optimizer": opt.count,
                      "rows": [row for _, rows in batches for row in rows],
                      "batches": copy.deepcopy(batches), "random": generator.random(),
                      "rate": opt.param_groups[0]["lr"]})
        return {"loss": 1., "source_tokens": n, "target_tokens": n}

    with patch.dict(sys.modules, {"torch": fake_torch}), \
            patch.object(backend, "load_pair", load), patch.object(backend, "optimizer", lambda *a: Optimizer()), \
            patch.object(backend, "update", update), \
            patch.object(training, "seed", generator.seed), \
            patch.object(training, "rng_state", generator.getstate), \
            patch.object(training, "restore_rng", generator.setstate), \
            contextlib.redirect_stdout(io.StringIO()):
        yield bound, selected, calls, loads


class ScheduleTests(unittest.TestCase):
    def test_five_trajectories_45_prefixes_and_full_horizon_accounting(self):
        plan = policy.plan()
        self.assertEqual((len(plan["trajectories"]), len(plan["milestones"])), (5, 45))
        self.assertEqual(plan["validation_generations_including_baseline"], 46 * 11332)
        self.assertEqual(schedule.EVALUATION_POINTS,
                         (10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000))
        for item in schedule.trajectories():
            self.assertGreaterEqual(item["updates"], 40000)
            self.assertEqual(item["points"], item["off_policy_examples"] + 2 * item["on_policy_examples"])
            self.assertEqual(schedule.accounting(item, item["updates"])["cursor"], item["examples"])
            previous_step, previous_groups = 0, 0
            for nominal, groups, step in schedule.boundaries(item["on_policy_percent"]):
                self.assertEqual(nominal % 5000, 0)
                self.assertEqual(schedule.groups_after(item, step), groups)
                self.assertEqual(schedule.groups_after(item, previous_step), previous_groups)
                last = policy.update_spec(item, step - 1)
                self.assertTrue(0 < last["examples"] <= 32)
                self.assertLess(nominal - groups * item["group_cost"], item["group_cost"])
                previous_step, previous_groups = step, groups
            for nominal in schedule.EVALUATION_POINTS:
                prefix = schedule.milestone(item["on_policy_percent"], nominal)
                self.assertEqual(prefix["updates"],
                                 schedule.boundaries(item["on_policy_percent"])[nominal // 5000 - 1][2])
                self.assertEqual(prefix["on_policy_examples"] * 100,
                                 prefix["examples"] * prefix["on_policy_percent"])
        self.assertEqual(plan["totals"]["points"], sum(x["points"] for x in schedule.trajectories()))
        self.assertGreater(sum(x["points"] for x in schedule.milestones()), plan["totals"]["points"])
        with self.assertRaises(ValueError):
            schedule.milestone(50, 5000)

    def test_new_bindings_select_design_and_legacy_bindings_are_not_upgraded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "grid"
            with patch.object(inputs, "separate_work"), patch.object(inputs, "inspect_pair", return_value={"fixed": True}), \
                    patch.object(inputs, "code_signature", return_value={"source": "provenance-only"}):
                bound = inputs.binding(root, Path(d) / "pair", policy.settings(), create=True)
                self.assertEqual(bound["training_design"], schedule.DESIGN)
                original = read_json(root / "binding.json")
                original.pop("training_design")
                atomic_json(root / "binding.json", original)
                before = (root / "binding.json").read_bytes()
                self.assertNotIn("training_design", inputs.binding(root, Path(d) / "pair", policy.settings()))
                self.assertEqual((root / "binding.json").read_bytes(), before)


class TrainingTests(unittest.TestCase):
    def test_full_horizon_finishes_and_retains_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = schedule.trajectory(100)
            with fake_training(root) as (bound, selected, calls, loads):
                owner = schedule.training_directory(root, item)
                value = controller.training_config(bound, item, selected)
                signatures = {"binding_sha256": sha256(root / "binding.json"),
                              "pilot_sha256": sha256(root / "pilot.json")}
                atomic_json(owner / "run.json", {"run_id": "fixture", "config": value,
                            "signatures": signatures, "runtime": selected["runtime"],
                            "initial_student": "original-fp32"})
                step = schedule.boundaries(100)[-2][2]
                state = {**engine.new_state(), **schedule.accounting(item, step)}
                checkpoints.save(owner, Model(step), SimpleNamespace(save_pretrained=lambda p: None), {
                    "step": step, "accounting": state, "optimizer": {"count": step},
                    "rng": random.Random(17).getstate(), "branch_rng": random.Random(19).getstate(),
                    "execution_config": value}, "synthetic test prefix")
                engine.train(root, bound, item, selected, until_points=2560000)
                result = read_json(owner / "training_summary.json")
                self.assertEqual((result["status"], result["step"], result["points"]),
                                 ("complete", 40448, 2560000))
                self.assertEqual(len(calls), item["updates"] - step)
                milestones.verify(root, bound, schedule.milestone(100, 2560000), selected)

    def test_restart_keeps_optimizer_rng_cursor_and_full_horizon_lr(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = schedule.trajectory(75)
            with fake_training(root) as (bound, selected, calls, loads):
                engine.train(root, bound, item, selected, session_updates=7, until_points=10000)
                engine.train(root, bound, item, selected, until_points=10000)
                split_calls = copy.deepcopy(calls)
                retained = schedule.evaluation_directory(root, schedule.milestone(75, 10000))
                source, record = milestones.verify(root, bound, schedule.milestone(75, 10000), selected)
                self.assertEqual(record["accounting"]["points"], 9996)
                self.assertEqual(pickle.loads((source / "training_state.pt").read_bytes())["optimizer"]["count"],
                                 schedule.milestone(75, 10000)["updates"])
                snapshot_hash = sha256(source / "COMPLETE.json")
                engine.train(root, bound, item, selected, until_points=20000)
                self.assertEqual(sha256(source / "COMPLETE.json"), snapshot_hash)
                self.assertEqual(len(list((root / "milestones").glob("p*"))), 2)
                self.assertEqual(len(checkpoints.completed(schedule.training_directory(root, item))), 2)
                self.assertFalse(any(retained.glob("evaluation*")))
                self.assertIsNone(loads[0])
                self.assertTrue(all(path is not None for path in loads[1:]))
                first = policy.update_spec(item, 0)
                self.assertEqual(calls[0]["rate"], bound["settings"]["learning_rate"]
                                 * policy.lr_factor(0, first["points"], item["points"]))
            other = root / "uninterrupted"
            with fake_training(other) as (bound, selected, calls, loads):
                engine.train(other, bound, item, selected, until_points=10000)
                self.assertEqual(calls, split_calls)

    def test_publication_failure_is_repaired_from_real_restart_without_retraining(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = schedule.trajectory(100)
            with fake_training(root) as (bound, selected, calls, loads):
                retain = milestones.retain
                def fail_at_milestone(*args):
                    if args[-1]["step"] == schedule.milestone(100, 10000)["updates"]:
                        raise OSError("simulated interrupted copy")
                    return retain(*args)
                with patch.object(milestones, "retain", side_effect=fail_at_milestone), self.assertRaises(OSError):
                    engine.train(root, bound, item, selected, until_points=10000)
                count = len(calls)
                engine.train(root, bound, item, selected, until_points=10000)
                self.assertEqual(len(calls), count)
                path, _ = milestones.verify(root, bound, schedule.milestone(100, 10000), selected)
                self.assertTrue(path.is_dir())
                (path / "model.safetensors").write_text("tampered")
                with self.assertRaises(ValueError):
                    milestones.verify(root, bound, schedule.milestone(100, 10000), selected)

    def test_rolling_checkpoints_are_not_retained_or_evaluated(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                item = schedule.trajectory(0)
                engine.train(root, bound, item, selected,
                             session_updates=schedule.boundaries(0)[0][2], until_points=10000)
                self.assertEqual(read_json(schedule.training_directory(root, item) / "training_summary.json")["points"], 5000)
                self.assertFalse((root / "milestones").exists())
                engine.train(root, bound, item, selected, until_points=10000)
                prefix = schedule.milestone(0, 10000)
                with patch.object(controller, "saved_pilot", return_value=selected):
                    self.assertEqual(evaluation.source(root, bound, prefix),
                                     schedule.evaluation_directory(root, prefix) / "checkpoint")
                with self.assertRaises(ValueError):
                    engine.check_state({**schedule.accounting(item, 1), "points": 99}, item)


class OrchestrationTests(unittest.TestCase):
    def test_evaluation_oom_uses_retained_snapshot_and_repairs_cached_pointer(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                engine.train(root, bound, schedule.trajectory(0), selected, until_points=10000)
                item = schedule.milestone(0, 10000)
                owner = schedule.evaluation_directory(root, item)
                snapshot = sha256(owner / "checkpoint/COMPLETE.json")
                attempts = []
                def call(operation, name, *extra):
                    self.assertEqual((operation, name), ("evaluate", item["id"]))
                    output = Path(extra[extra.index("--output") + 1])
                    attempts.append(output)
                    output.mkdir()
                    if len(attempts) == 1:
                        (output / "partial-output").write_text("preserve failed attempt")
                        raise recovery.CudaOOM({"operation":"evaluate", "cell":item["id"]})
                    (output / "predictions.jsonl").write_text("mock predictions\n")
                    atomic_json(output / "metrics.json", {
                        "status":"complete", "n":11332, "split":"validation",
                        "scores":{"rouge2":0.}, "predictions_sha256":sha256(output / "predictions.jsonl")})
                    atomic_json(output / "COMPLETE.json", {
                        "identity":evaluation.identity(root, bound, item, selected), "n":11332,
                        "files":tree_manifest(output)})
                    return True
                with patch.object(controller, "saved_pilot", return_value=selected):
                    runner = SimpleNamespace(call=call)
                    self.assertTrue(controller.ensure_evaluation(root, bound, item, selected, runner))
                    self.assertFalse((owner / "latest_evaluation.json").exists())
                    self.assertTrue(controller.ensure_evaluation(root, bound, item, selected, runner))
                    self.assertEqual(read_json(owner / "latest_evaluation.json")["path"], str(attempts[1]))
                    self.assertEqual(len(attempts), 2)
                    self.assertEqual(recovery.evaluation_microbatch(root, bound, item, selected), 16)
                self.assertTrue((attempts[0] / "partial-output").exists())
                self.assertEqual(sha256(owner / "checkpoint/COMPLETE.json"), snapshot)

    def test_worker_cli_routes_training_and_evaluation_to_distinct_owners(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads), \
                    patch.object(inputs, "separate_work"), patch.object(inputs, "binding", return_value=bound), \
                    patch.object(controller, "saved_pilot", return_value=selected), \
                    patch("sftlib.core.configure_local_storage"), \
                    patch.object(recovery, "training_microbatch", return_value=32), \
                    patch.object(recovery, "evaluation_microbatch", return_value=32):
                prefix = ["distill_grid.py", "--work-dir", str(root), "--sft-work", str(root / "pair")]
                with patch.object(sys, "argv", [*prefix, "_train", "--cell", "on000", "--until-points", "10000"]), \
                        patch.object(engine, "train") as train:
                    distill_grid.main()
                    self.assertEqual(train.call_args.args[2], schedule.trajectory(0))
                    self.assertEqual(train.call_args.kwargs["until_points"], 10000)
                item = schedule.milestone(0, 10000)
                output = schedule.evaluation_directory(root, item) / "evaluation"
                with patch.object(sys, "argv", [*prefix, "_evaluate", "--cell", item["id"],
                                               "--output", str(output)]), \
                        patch.object(evaluation, "evaluate") as evaluate:
                    distill_grid.main()
                    self.assertEqual(evaluate.call_args.args[2], item)
                    self.assertEqual(read_json(output.parent / "latest_evaluation.json")["path"], str(output))

    def test_controller_trains_each_mixture_once_and_evaluates_only_declared_prefixes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            done, evaluated, trained = set(), set(), []
            def train(root, item, selected, runner, until_points=None):
                self.assertEqual(item["budget_points"], 2560000)
                trained.append((item["id"], until_points))
                done.add(schedule.milestone(item["on_policy_percent"], until_points)["id"])
                return True
            def evaluate(root, bound, item, selected, runner):
                self.assertIn(item["id"], done)
                evaluated.add(item["id"])
                return True
            summary = {"complete_cells": 45, "complete_trajectories": 5}
            with patch.object(controller, "check_completed_training", side_effect=lambda r,b,i,s: i["id"] in done), \
                    patch.object(controller, "train_with_recovery", train), \
                    patch.object(controller, "ensure_evaluation", evaluate), \
                    patch.object(reporting, "summary", return_value=summary), \
                    contextlib.redirect_stdout(io.StringIO()):
                controller.run_trajectories(root, {}, {}, None, SimpleNamespace(requested=False))
                self.assertEqual(len(trained), 45)
                self.assertEqual(len({x[0] for x in trained}), 5)
                self.assertEqual(evaluated, {x["id"] for x in schedule.milestones()})
                controller.run_trajectories(root, {}, {}, None, SimpleNamespace(requested=False))
                self.assertEqual(len(trained), 45)
                self.assertEqual(read_json(root / "status.json")["status"], "complete")

    def test_oom_retry_keeps_milestone_target_and_uses_trajectory_restart(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = schedule.trajectory(50)
            calls = []
            def call(*args):
                calls.append(args)
                if len(calls) == 1:
                    raise recovery.CudaOOM({"operation": "train", "cell": item["id"]})
                return True
            with patch.object(recovery, "training_microbatch", side_effect=[32, 16]), \
                    patch.object(recovery, "checkpoint_for_retry", return_value=(Path("real-restart"), {"step": 10})), \
                    patch.object(recovery, "lower") as lower:
                self.assertTrue(recovery.train_with_recovery(
                    root, item, {"microbatch":32}, SimpleNamespace(call=call), until_points=40000))
            self.assertTrue(all(args[-2:] == ("--until-points", 40000) for args in calls))
            self.assertEqual(lower.call_args.kwargs["checkpoint"], "real-restart")

    def test_summary_and_estimate_do_not_sum_prefix_training_as_independent_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                item = schedule.trajectory(0)
                engine.train(root, bound, item, selected, until_points=20000)
                result = reporting.summary(root)
                self.assertEqual(len(result["rows"]), 45)
                self.assertEqual(result["rows"][0]["status"], "trained")
                self.assertEqual(result["rows"][1]["status"], "trained")
                actual = read_json(schedule.training_directory(root, item) / "training_summary.json")
                self.assertEqual(result["committed_training_seconds"], actual["training_seconds"])
                self.assertLess(result["committed_training_seconds"],
                                sum(row["training_seconds"] or 0 for row in result["rows"]))
            atomic_json(root / "baseline/latest_evaluation.json", {"path": str(root / "baseline/evaluation")})
            atomic_json(root / "baseline/evaluation/COMPLETE.json", {"identity": "mock"})
            branch = {"mean_update_seconds": 1., "checkpoint_seconds": .1, "wall_seconds": 5.,
                      "updates": [{"training_seconds":1.}], "resumed_update":{"training_seconds":1.}}
            selected = {"microbatch": 32, "off_policy": branch, "on_policy": branch}
            with patch.object(evaluation, "verify", return_value={
                    "n":11332, "status":"complete", "evaluation_seconds":2., "wall_seconds":3.}):
                estimate = reporting.estimate(root, selected, policy.settings())
            self.assertEqual(len(estimate["trajectories"]), 5)
            self.assertEqual(sum(x["evaluations"] for x in estimate["trajectories"]), 45)
            self.assertEqual(sum(x["points"] for x in estimate["trajectories"]), policy.plan()["totals"]["points"])


if __name__ == "__main__":
    unittest.main()
