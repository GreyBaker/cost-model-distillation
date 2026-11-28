"""Focused continuous-trajectory regressions; fake tensors/optimizer, no ML or GPU."""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
import pickle
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gridlib import (backend, controller, engine, evaluation, inputs, milestones,
                     policy, recovery, reporting, schedule, trajectory_reporting, audit)
from gridlib.shared import atomic_json, checkpoints, read_json, sha256, tree_manifest
from sftlib import training
import distill_grid
import run_pipeline


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
    atomic_json(root / "evaluation_pilot.json", {"mock": True, "batch_size": 32})
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
            patch("gridlib.evalpilot.saved", return_value={"batch_size": 32}), \
            contextlib.redirect_stdout(io.StringIO()):
        yield bound, selected, calls, loads


def seed_checkpoint(root, bound, selected, percent, nominal):
    """Development-only synthetic history, never used by the real workflow."""
    item = schedule.trajectory(percent)
    owner = schedule.training_directory(root, item)
    config = controller.training_config(bound, item, selected)
    signatures = {"binding_sha256": sha256(root / "binding.json"),
                  "pilot_sha256": sha256(root / "pilot.json")}
    atomic_json(owner / "run.json", {"run_id": f"fixture-{percent}", "config": config,
                "signatures": signatures, "runtime": selected["runtime"],
                "initial_student": "original-fp32"})
    step = schedule.boundaries(percent)[nominal // 5000 - 1][2]
    state = {**engine.new_state(), **schedule.accounting(item, step)}
    return checkpoints.save(owner, Model(step), SimpleNamespace(save_pretrained=lambda p: None), {
        "step": step, "accounting": state, "optimizer": {"count": step},
        "rng": random.Random(17).getstate(), "branch_rng": random.Random(19).getstate(),
        "execution_config": config}, "synthetic test prefix")


class ScheduleTests(unittest.TestCase):
    def test_nominal_160k_lr_warmup_plateau_and_cooldown(self):
        scope = schedule.campaign()
        self.assertEqual((scope["lr_warmup_points"], scope["lr_cooldown_begin_points"],
                          scope["lr_cooldown_end_points"]), (8000, 120000, 160000))
        # lr_factor uses the update's midpoint, including the final short update.
        for midpoint, expected in ((4000, .5), (8000, 1.), (80000, 1.),
                                   (120000, 1.), (140000, .5), (159990, .00025)):
            self.assertAlmostEqual(policy.lr_factor(midpoint - 10, 20, 160000), expected)
        for item in schedule.trajectories():
            first = policy.update_spec(item, 0)
            factor = policy.lr_factor(0, first["points"], item["budget_points"])
            self.assertAlmostEqual(factor, first["points"] / 2 / 8000)
            last = policy.update_spec(item, item["updates"] - 1)
            previous = schedule.accounting(item, item["updates"] - 1)["points"]
            self.assertLess(policy.lr_factor(previous, last["points"], item["budget_points"]), .002)

    def test_old_long_horizon_binding_is_rejected_before_in_place_writes(self):
        original_design = {
            "kind": "continuous-point-milestones-v1", "final_points": 2560000,
            "evaluation_points": [10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000],
            "recovery_every_points": 5000, "recovery_retention": 2,
            "boundary_policy": "complete_exact_mixture_groups_at_or_below_nominal_points",
            "update_policy": "batch_32_except_short_update_at_recovery_boundary",
            "lr_horizon": "full_trajectory_spendable_points",
        }
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(root / "binding.json", {"training_design": original_design,
                        "protocol": policy.PROTOCOL, "settings": policy.settings()})
            atomic_json(root / "trajectories/on000/run.json", {"old_history": "preserve"})
            before = tree_manifest(root)
            for operation in ("run", "estimate", "summary", "audit"):
                with self.subTest(operation=operation), patch.object(sys, "argv", [
                        "distill_grid.py", "--work-dir", str(root), operation]), \
                        patch("sftlib.core.configure_local_storage",
                              side_effect=AssertionError("No setup or cache writes")), \
                        self.assertRaisesRegex(ValueError, "long-horizon checkpoints cannot be relabeled"):
                    distill_grid.main()
                self.assertEqual(tree_manifest(root), before)

    def test_five_trajectories_25_prefixes_and_intended_160k_horizon(self):
        plan = policy.plan()
        self.assertEqual((len(plan["trajectories"]), len(plan["milestones"])), (5, 25))
        self.assertEqual(plan["validation_generations_including_baseline"], 26 * 11332)
        self.assertEqual(schedule.evaluation_points(), (10000, 20000, 40000, 80000, 160000))
        self.assertEqual(plan["campaign"]["end_points"], 160000)
        self.assertEqual(plan["campaign"]["lr_horizon_nominal_points"], 160000)
        self.assertEqual(plan["totals"]["budget_points"], 800000)
        self.assertEqual(plan["totals"]["points"], 799998)
        self.assertEqual(plan["totals"]["updates"], 17792)
        self.assertEqual(plan["totals"]["examples"], 566094)
        self.assertEqual(schedule.EVALUATION_POINTS,
                         (10000, 20000, 40000, 80000, 160000))
        for item in schedule.trajectories():
            self.assertEqual(item["budget_points"], 160000)
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
        self.assertEqual(plan["totals"]["points"],
                         sum(schedule.milestone(p, 160000)["points"] for p in schedule.PERCENTAGES))
        self.assertGreater(sum(x["points"] for x in schedule.milestones()), plan["totals"]["points"])
        with self.assertRaises(ValueError):
            schedule.milestone(50, 5000)
        self.assertEqual(len(policy.plan(80000)["milestones"]), 20)
        self.assertEqual(policy.plan(80000)["training_design"], plan["training_design"])
        for invalid in (False, 5000, 320000, 640000, 2560000):
            with self.assertRaises(ValueError):
                schedule.campaign(invalid)

    def test_new_bindings_select_design_and_old_bindings_are_rejected_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "grid"
            with patch.object(inputs, "separate_work"), patch.object(inputs, "inspect_pair", return_value={"fixed": True}), \
                    patch.object(inputs, "code_signature", return_value={"source": "provenance-only"}):
                bound = inputs.binding(root, Path(d) / "pair", policy.settings(), create=True)
                self.assertEqual(bound["training_design"], schedule.DESIGN)
                # Smaller operational scopes never change this recipe's 160K LR horizon.
                binding_hash = sha256(root / "binding.json")
                for end_points in (40000, 80000, 160000):
                    self.assertEqual(policy.plan(end_points)["training_design"], bound["training_design"])
                    inputs.binding(root, Path(d) / "pair", policy.settings())
                    self.assertEqual(sha256(root / "binding.json"), binding_hash)
                original = read_json(root / "binding.json")
                original.pop("training_design")
                atomic_json(root / "binding.json", original)
                before = (root / "binding.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "NEW grid directory"):
                    inputs.binding(root, Path(d) / "pair", policy.settings())
                self.assertEqual((root / "binding.json").read_bytes(), before)

    def test_obsolete_grid_is_rejected_by_every_operational_cli_before_any_writes(self):
        operations = [
            ["estimate"], ["summary"], ["audit"],
            ["run"], ["run", "--accept-estimate"], ["_runtime"],
            ["_pilot"], ["_evaluation_pilot"],
            ["_train", "--cell", "p05000_on000"], ["_evaluate", "--cell", "p05000_on000"],
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(root / "binding.json", {"protocol":policy.PROTOCOL, "settings":policy.settings()})
            atomic_json(root / "status.json", {"status":"complete", "complete_cells":25})
            (root / "summary.csv").write_text("archived results\n")
            before = tree_manifest(root)
            for operation in operations:
                with self.subTest(operation=operation), patch.object(sys, "argv", [
                        "distill_grid.py", "--work-dir", str(root), *operation]), \
                        patch("sftlib.core.configure_local_storage",
                              side_effect=AssertionError("No cache/environment writes allowed")), \
                        self.assertRaisesRegex(ValueError, "NEW grid directory"):
                    distill_grid.main()
                self.assertEqual(tree_manifest(root), before)
            with self.assertRaisesRegex(ValueError, "NEW grid directory"):
                reporting.summary(root)
            with self.assertRaisesRegex(ValueError, "NEW grid directory"):
                reporting.estimate(root, {}, {})
            self.assertEqual(tree_manifest(root), before)

    def test_old_cell_api_settings_and_unbound_directory_adoption_are_removed(self):
        self.assertFalse(hasattr(policy, "cell"))
        self.assertFalse(hasattr(policy, "cells"))
        self.assertFalse(hasattr(policy, "legacy_plan"))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = {"id":"p05000_on000", "on_policy_percent":0, "budget_points":5000, "updates":157}
            with self.assertRaises(ValueError):
                policy.update_spec(old, 0)
            with self.assertRaises(ValueError):
                engine.train(root, {"training_design":schedule.DESIGN}, old, {})
            with self.assertRaises(ValueError):
                evaluation.source(root, {"training_design":schedule.DESIGN}, old)
            atomic_json(root / "old-settings.json", {**policy.settings(), "save_every_updates":50})
            with self.assertRaisesRegex(ValueError, "settings fields"):
                policy.settings(root / "old-settings.json")
            before = tree_manifest(root)
            with self.assertRaisesRegex(ValueError, "NEW grid directory"):
                schedule.validate_root(root, allow_new=True)
            self.assertEqual(tree_manifest(root), before)

    def test_outer_controllers_reject_old_grid_before_storage_or_workers(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod-external-teacher"))
        from external_workflow import controller as external_controller
        with tempfile.TemporaryDirectory() as d:
            disk = Path(d)
            work, grid = disk / "external", disk / "old-grid"
            atomic_json(grid / "binding.json", {"old_grid":True})
            before = tree_manifest(disk)
            forbidden = AssertionError("No setup/runtime/worker calls for an unsupported grid")
            args = SimpleNamespace(pod_disk=disk, work_dir=work, grid_work=grid, student="public")
            with patch.object(external_controller, "operational_paths", return_value=(disk, work, grid)), \
                    patch.object(external_controller, "configure_local_storage", side_effect=forbidden), \
                    patch.object(external_controller, "runtime", side_effect=forbidden), \
                    self.assertRaisesRegex(ValueError, "NEW grid directory"):
                external_controller.run(args)
            with patch.object(run_pipeline, "ROOT", disk / "code/runpod-gkd"), \
                    patch.object(run_pipeline, "SFT_CODE", disk / "code/runpod-sft"), \
                    patch.dict("os.environ", {"SFT_WORK":str(disk / "sft"), "GKD_WORK":str(grid)}), \
                    patch.object(run_pipeline, "configure_local_storage", side_effect=forbidden), \
                    patch.object(run_pipeline, "invoke", side_effect=forbidden), \
                    self.assertRaisesRegex(ValueError, "NEW grid directory"):
                run_pipeline.run(disk, reuse_sft=True)
            self.assertEqual(tree_manifest(disk), before)


class TrainingTests(unittest.TestCase):
    def test_160k_endpoint_keeps_full_state_and_can_reload_permanent_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root, other = Path(d) / "split", Path(d) / "uninterrupted"
            item = schedule.trajectory(75)
            with fake_training(root) as (bound, selected, calls, loads):
                seed_checkpoint(root, bound, selected, 75, 155000)
                binding_hash = sha256(root / "binding.json")
                engine.train(root, bound, item, selected)
                result = read_json(schedule.training_directory(root, item) / "training_summary.json")
                self.assertEqual(result["points"], schedule.milestone(75, 160000)["points"])
                self.assertEqual(result["status"], "complete")
                checkpoint, record = milestones.verify(root, bound, schedule.milestone(75, 160000), selected)
                state = pickle.loads((checkpoint / "training_state.pt").read_bytes())
                self.assertTrue({"optimizer", "rng", "branch_rng", "accounting", "execution_config"} <= state.keys())
                snapshot_hash = tree_manifest(checkpoint)
                # Even if the rolling restarts are unavailable, the permanent endpoint is restartable.
                shutil.rmtree(schedule.training_directory(root, item) / "checkpoints")
                count = len(calls)
                engine.train(root, bound, item, selected)
                self.assertEqual(len(calls), count)
                self.assertEqual(loads[-1], checkpoint)
                self.assertEqual(tree_manifest(checkpoint), snapshot_hash)
                self.assertEqual(sha256(root / "binding.json"), binding_hash)
                split_calls = copy.deepcopy(calls)
            with fake_training(other) as (bound, selected, calls, loads):
                seed_checkpoint(other, bound, selected, 75, 155000)
                engine.train(other, bound, item, selected)
                self.assertEqual(calls, split_calls)

    def test_worker_already_beyond_endpoint_does_not_load_gpu_or_rewrite_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                seed_checkpoint(root, bound, selected, 50, 85000)
                before = tree_manifest(root)
                engine.train(root, bound, schedule.trajectory(50), selected, until_points=80000)
                self.assertEqual(tree_manifest(root), before)
                self.assertFalse(calls)
                self.assertFalse(loads)

    def test_exact_endpoint_restart_repairs_snapshot_without_retraining(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                original = seed_checkpoint(root, bound, selected, 50, 160000)
                before = tree_manifest(original)
                engine.train(root, bound, schedule.trajectory(50), selected)
                snapshot, record = milestones.verify(root, bound, schedule.milestone(50, 160000), selected)
                self.assertEqual(record["accounting"]["points"], 159999)
                self.assertEqual(tree_manifest(snapshot), before)
                self.assertFalse(calls)
                self.assertEqual(loads, [original])
                self.assertTrue(read_json(root / "trajectories/on050/training_summary.json")["target_reached"])

    def test_already_advanced_checkpoint_still_requires_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                checkpoint = seed_checkpoint(root, bound, selected, 0, 85000)
                (checkpoint / "training_state.pt").write_bytes(b"corrupt")
                before = tree_manifest(root)
                with self.assertRaisesRegex(ValueError, "integrity failure"):
                    controller.run_trajectories(root, bound, selected, None, SimpleNamespace(requested=False), 80000)
                self.assertEqual(tree_manifest(root), before)
                self.assertFalse(calls)
                self.assertFalse(loads)

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
                engine.train(root, bound, item, selected)
                result = read_json(owner / "training_summary.json")
                self.assertEqual((result["status"], result["step"], result["points"]),
                                 ("complete", 2528, 160000))
                self.assertEqual(len(calls), item["updates"] - step)
                milestones.verify(root, bound, schedule.milestone(100, 160000), selected)

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
                                 * policy.lr_factor(0, first["points"], item["budget_points"]))
            other = root / "uninterrupted"
            with fake_training(other) as (bound, selected, calls, loads):
                engine.train(other, bound, item, selected, until_points=10000)
                self.assertEqual(calls, split_calls)

    def test_real_engine_uses_nominal_160k_lr_for_every_mixture(self):
        for percent in schedule.PERCENTAGES:
            with self.subTest(percent=percent), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                item = schedule.trajectory(percent)
                with fake_training(root) as (bound, selected, calls, loads):
                    engine.train(root, bound, item, selected, session_updates=1, until_points=10000)
                    spec = policy.update_spec(item, 0)
                    self.assertEqual(calls[0]["rate"], bound["settings"]["learning_rate"]
                                     * policy.lr_factor(0, spec["points"], 160000))
                    self.assertNotEqual(calls[0]["rate"], bound["settings"]["learning_rate"]
                                        * policy.lr_factor(0, spec["points"], 2560000))

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
    def test_single_cli_run_binds_before_cache_setup_without_manual_prerequisites(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "new-grid"
            order = []
            def setup(path):
                self.assertTrue((path / "binding.json").is_file())
                order.append("storage")
            def run(path, bound, settings_path, end_points):
                self.assertEqual(bound["training_design"], schedule.DESIGN)
                self.assertEqual(end_points, 160000)
                order.append("run")
            with patch.object(sys, "argv", ["distill_grid.py", "--work-dir", str(root), "run"]), \
                    patch.object(inputs, "separate_work"), \
                    patch.object(inputs, "inspect_pair", return_value={"kind": "test-pair"}), \
                    patch.object(inputs, "code_signature", return_value={"code": "provenance"}), \
                    patch("sftlib.core.configure_local_storage", side_effect=setup), \
                    patch.object(controller, "run", side_effect=run):
                distill_grid.main()
            self.assertEqual(order, ["storage", "run"])
            self.assertTrue((root / "binding.json").exists())

    def test_manual_bind_and_pilot_commands_are_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "absent"
            for operation in ("bind", "pilot"):
                with self.subTest(operation=operation), patch.object(sys, "argv", [
                        "distill_grid.py", "--work-dir", str(root), operation]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    distill_grid.main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(root.exists())

    def test_single_grid_run_automatically_calibrates_baseline_trains_and_audits_then_reuses_fit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound = {"training_design": schedule.DESIGN, "settings": policy.settings(),
                     "protocol": policy.PROTOCOL, "inputs": {"sft_work": "pair"}}
            atomic_json(root / "binding.json", bound)
            runtime, events = {"mock": True}, []
            def call(runner, operation, name, *extra):
                if operation == "runtime":
                    events.append("runtime")
                    atomic_json(root / "runtime.json", runtime)
                elif operation == "pilot":
                    events.append("fit")
                    batch = int(extra[extra.index("--microbatch") + 1])
                    online = bool(int(extra[extra.index("--online") + 1]))
                    output = Path(extra[extra.index("--output") + 1])
                    atomic_json(output, {"microbatch": batch, "online": online, "status": "pass",
                                "runtime": runtime, "binding_sha256": sha256(root / "binding.json"),
                                "mean_update_seconds": 32 / batch})
                else:
                    raise AssertionError(operation)
                return True
            def mark(name, result):
                def execute(*args, **kwargs):
                    events.append(name)
                    return result
                return execute
            def train(*args, **kwargs):
                events.append("train")
                atomic_json(root / "status.json", {"status": "complete", "complete_cells": 25})
            with patch.object(controller.Runner, "call", call), \
                    patch.object(controller.evalpilot, "ensure", side_effect=mark("evaluation_calibration", True)), \
                    patch.object(controller, "ensure_evaluation", side_effect=mark("baseline", True)), \
                    patch.object(reporting, "estimate", side_effect=mark("estimate", {"mock_estimate": True})), \
                    patch.object(controller, "run_trajectories", side_effect=train), \
                    patch.object(audit, "audit", side_effect=mark("audit", {"status": "pass"})), \
                    contextlib.redirect_stdout(io.StringIO()):
                controller.run(root, bound, Path("settings.json"))
                self.assertEqual(events.count("fit"), 2 * len(bound["settings"]["pilot_microbatches"]))
                self.assertEqual(events[-5:], ["evaluation_calibration", "baseline", "estimate", "train", "audit"])
                fit_hash = sha256(root / "pilot.json")
                events.clear()
                controller.run(root, bound, Path("settings.json"))
                self.assertNotIn("fit", events)
                self.assertEqual(sha256(root / "pilot.json"), fit_hash)
                self.assertEqual(events[-5:], ["evaluation_calibration", "baseline", "estimate", "train", "audit"])
            self.assertEqual(read_json(root / "status.json")["audit_status"], "pass")
            self.assertEqual(read_json(root / "audit.json")["status"], "pass")

    def test_interrupted_automatic_calibration_does_not_start_training_or_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound = {"training_design": schedule.DESIGN, "inputs": {"sft_work": "pair"}}
            with patch.object(controller, "calibrate", return_value=None), \
                    patch.object(controller, "run_trajectories", side_effect=AssertionError("No training")), \
                    patch.object(audit, "audit", side_effect=AssertionError("No audit")):
                controller.run(root, bound, Path("settings.json"))
            self.assertEqual(read_json(root / "status.json")["status"], "paused")

    def test_single_run_does_not_mark_failed_audit_as_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bound = {"training_design": schedule.DESIGN, "settings": policy.settings(),
                     "inputs": {"sft_work": "pair"}}
            def trained(*args, **kwargs):
                atomic_json(root / "status.json", {"status": "complete"})
            with patch.object(controller, "calibrate", return_value={"mock": True}), \
                    patch.object(reporting, "estimate", return_value={}), \
                    patch.object(controller, "run_trajectories", side_effect=trained), \
                    patch.object(audit, "audit", return_value={"status": "fail"}), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "Final grid integrity audit did not pass"):
                controller.run(root, bound, Path("settings.json"))
            self.assertNotEqual(read_json(root / "status.json")["status"], "complete")
            self.assertEqual(read_json(root / "audit.json")["status"], "fail")

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
                with patch.object(sys, "argv", [*prefix, "--end-points", "80000", "_train", "--cell", "on000"]), \
                        patch.object(engine, "train") as train:
                    distill_grid.main()
                    self.assertEqual(train.call_args.kwargs["until_points"], 80000)
                command = controller.command(root, root / "pair", Path("settings.json"), "_train",
                                             "--cell", "on000", end_points=80000)
                self.assertEqual(command[command.index("--end-points") + 1], "80000")
                self.assertLess(command.index("--end-points"), command.index("_train"))
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
                self.assertEqual(item["budget_points"], 160000)
                trained.append((item["id"], until_points))
                done.add(schedule.milestone(item["on_policy_percent"], until_points)["id"])
                return True
            def evaluate(root, bound, item, selected, runner):
                self.assertIn(item["id"], done)
                evaluated.add(item["id"])
                return True
            summary = {"complete_cells": 25, "complete_trajectories": 5,
                       "campaign_complete": True, "skipped_trajectories": []}
            with patch.object(controller, "check_completed_training", side_effect=lambda r,b,i,s: i["id"] in done), \
                    patch.object(controller, "train_with_recovery", train), \
                    patch.object(controller, "ensure_evaluation", evaluate), \
                    patch.object(reporting, "summary", return_value=summary), \
                    contextlib.redirect_stdout(io.StringIO()):
                controller.run_trajectories(root, {"training_design":schedule.DESIGN}, {}, None,
                                            SimpleNamespace(requested=False))
                self.assertEqual(len(trained), 25)
                self.assertEqual(len({x[0] for x in trained}), 5)
                self.assertEqual(evaluated, {x["id"] for x in schedule.active_milestones()})
                controller.run_trajectories(root, {"training_design":schedule.DESIGN}, {}, None,
                                            SimpleNamespace(requested=False))
                self.assertEqual(len(trained), 25)
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

    def test_in_place_controller_skips_advanced_mixture_and_missing_past_but_reuses_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            done = {schedule.milestone(25, 20000)["id"]}
            trained, evaluated = [], []
            def restart(root, bound, trajectory, selected):
                percent = trajectory["on_policy_percent"]
                if percent == 0:
                    return Path("later"), {"step": schedule.milestone(0, 160000)["updates"]}
                if percent == 25:
                    return Path("partial"), {"step": schedule.boundaries(25)[4][2]}  # 25K
                return None, None
            def train(root, item, selected, runner, until_points=None):
                trained.append((item["on_policy_percent"], until_points))
                done.add(schedule.milestone(item["on_policy_percent"], until_points)["id"])
                return True
            def evaluate(root, bound, item, selected, runner):
                self.assertIn(item["id"], done)
                evaluated.append(item["id"])
                return True
            summary = {"complete_cells": 15, "complete_trajectories": 3,
                       "campaign_complete": True, "skipped_trajectories": ["on000"]}
            with patch.object(milestones, "latest_restart", side_effect=restart), \
                    patch.object(controller, "check_completed_training", side_effect=lambda r,b,i,s: i["id"] in done), \
                    patch.object(controller, "train_with_recovery", side_effect=train), \
                    patch.object(controller, "ensure_evaluation", side_effect=evaluate), \
                    patch.object(reporting, "summary", return_value=summary), \
                    contextlib.redirect_stdout(io.StringIO()):
                controller.run_trajectories(root, {"training_design": schedule.DESIGN}, {}, None,
                                            SimpleNamespace(requested=False), 80000)
            self.assertEqual(len(trained), 14)
            self.assertEqual(len(evaluated), 15)
            self.assertFalse(any(percent == 0 or (percent == 25 and point <= 25000) for percent, point in trained))
            self.assertIn(schedule.milestone(25, 20000)["id"], evaluated)
            self.assertNotIn(schedule.milestone(25, 10000)["id"], evaluated)
            self.assertTrue(all(point <= 80000 for _, point in trained))
            status = read_json(root / "status.json")
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["skipped_trajectories"], ["on000"])

    def test_exceeded_mixtures_finish_without_fake_evaluations_and_audit_reports_skips(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                for percent in schedule.PERCENTAGES:
                    seed_checkpoint(root, bound, selected, percent, 85000)
                protected = tree_manifest(root)
                with patch.object(controller, "train_with_recovery", side_effect=AssertionError("No retraining")), \
                        patch.object(controller, "ensure_evaluation", side_effect=AssertionError("No evaluation backlog")):
                    controller.run_trajectories(root, bound, selected, None, SimpleNamespace(requested=False), 80000)
                result = read_json(root / "summary.json")
                self.assertTrue(result["campaign_complete"])
                self.assertEqual(result["complete_cells"], 0)
                self.assertEqual(result["complete_trajectories"], 0)
                self.assertEqual(len(result["skipped_trajectories"]), 5)
                self.assertEqual(len(result["skipped_milestones"]), 20)
                self.assertTrue(all(row["rouge2"] is None for row in result["rows"]))
                self.assertEqual(read_json(root / "status.json")["status"], "complete")
                atomic_json(root / "baseline/latest_evaluation.json", {"path": str(root / "baseline/evaluation")})
                with patch("gridlib.audit.saved_pilot", return_value=selected), \
                        patch.object(evaluation, "identity", return_value={}), \
                        patch.object(evaluation, "verify"), patch.object(audit, "prediction_audit", return_value={}):
                    report = audit.audit(root, bound, 80000)
                self.assertEqual(report["status"], "pass_with_skips")
                self.assertFalse(report["cells"])
                self.assertEqual(len(report["skipped"]), 20)
                after = tree_manifest(root)
                self.assertTrue(all(after[name] == info for name, info in protected.items()))

    def test_summary_and_estimate_do_not_sum_prefix_training_as_independent_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with fake_training(root) as (bound, selected, calls, loads):
                item = schedule.trajectory(0)
                engine.train(root, bound, item, selected, until_points=20000)
                result = reporting.summary(root)
                self.assertEqual(len(result["rows"]), 25)
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
            self.assertEqual(sum(x["evaluations"] for x in estimate["trajectories"]), 25)
            self.assertEqual(sum(x["points"] for x in estimate["trajectories"]), policy.plan()["totals"]["points"])
            self.assertTrue(all(row["recovery_saves_including_initial"] == 33 for row in estimate["trajectories"]))
            self.assertEqual(estimate["campaign"]["end_points"], 160000)
            with patch.object(evaluation, "verify", return_value={
                    "n":11332, "status":"complete", "evaluation_seconds":2., "wall_seconds":3.}):
                shorter = reporting.estimate(root, selected, policy.settings(), 80000)
            self.assertEqual(sum(x["evaluations"] for x in shorter["trajectories"]), 20)
            self.assertEqual(shorter["campaign"]["lr_horizon_nominal_points"], 160000)


if __name__ == "__main__":
    unittest.main()
