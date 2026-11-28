"""Standard-library workflow contract checks. No network, ML imports, GPU or writes."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan


class SchemeTests(unittest.TestCase):
    def test_restored_student_default_and_no_local_sft(self):
        result = plan.build()
        self.assertEqual(result["student_mode"], "restored")
        self.assertEqual(result["status"], "implemented_gpu_validation_required")
        self.assertEqual(result["local_sft_updates"], 0)
        self.assertFalse(result["periodic_greedy_monitor"])
        self.assertIn("compare_legacy_and_native_student_logits_gradients_and_parameter_aliases", result["stages"])
        self.assertNotIn("download_and_verify_locked_public_student_sources", result["stages"])

    def test_public_student_is_explicit_separate_mode(self):
        result = plan.build("public")
        self.assertEqual(result["student"]["repo_id"], "yoshitomo-matsubara/t5_small-xsum")
        self.assertIn("download_and_verify_locked_public_student_sources", result["stages"])
        self.assertNotIn("compare_legacy_and_native_student_logits_gradients_and_parameter_aliases", result["stages"])
        with self.assertRaises(ValueError):
            plan.build("automatic")

    def test_grid_totals_count_continuous_training_once_not_each_prefix(self):
        totals = plan.build()["grid_totals"]
        self.assertEqual(totals["cells"], 25)
        self.assertEqual(totals["training_trajectories"], 5)
        # Cross-check actual existing engine policy, not just copied constants.
        sys.path.insert(0, str(plan.ROOT.parent / "runpod-gkd"))
        from gridlib import schedule
        self.assertEqual(plan.build()["grid_cells"], schedule.active_milestones())
        self.assertEqual(plan.build()["grid_trajectories"],
                         [schedule.milestone(p, 160000) for p in schedule.PERCENTAGES])
        for key in ("points", "examples", "updates", "unspent_points"):
            self.assertEqual(totals[key], sum(schedule.milestone(p, 160000)[key] for p in schedule.PERCENTAGES))
        campaign = plan.build()["campaign"]
        self.assertEqual((campaign["lr_horizon_nominal_points"], campaign["lr_warmup_points"],
                          campaign["lr_cooldown_begin_points"], campaign["lr_cooldown_end_points"]),
                         (160000, 8000, 120000, 160000))

    def test_wrong_lr_recipe_is_rejected(self):
        locked = plan.read(plan.ROOT / "models.lock.json")
        for key, value in (("training_design", "continuous-point-milestones-v1"),
                           ("lr_horizon", "full_trajectory_spendable_points"),
                           ("lr_warmup_points", 16000), ("lr_cooldown_begin_points", 240000),
                           ("lr_cooldown_end_points", 320000)):
            changed = plan.read(plan.ROOT / "scheme.json")
            changed["grid"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Grid.*(recipe|schedule) changed"):
                plan.validate(changed, locked)

    def test_single_command_verifies_and_reports_without_manual_workflow_pauses(self):
        stages = plan.build()["stages"]
        self.assertLess(stages.index("full_11332_teacher_and_student_validation_and_paired_difference"),
                        stages.index("publish_honest_external_pair_manifest_atomically"))
        self.assertLess(stages.index("display_informational_quality_and_verify_technical_integrity"),
                        stages.index("publish_honest_external_pair_manifest_atomically"))
        self.assertNotIn("disposable_gkd_update_save_reload_resume_smoke_test", stages)
        self.assertEqual(plan.build()["development"], {
            "smoke_command": "external_teacher.py dev-smoke", "included_in_standard_run": False})
        self.assertLess(stages.index("print_measured_estimate_and_continue_automatically"),
                         stages.index("train_5_continuous_mixtures_through_160k_with_25_retained_evaluated_prefixes"))
        self.assertEqual(plan.build()["execution"], {
            "command": "external_teacher.py run", "manual_stage_or_pilot_commands": False,
            "manual_approval_pauses": False, "automatic_calibration": True})

    def test_pins_weights_and_equal_public_tokenizer_file_identities(self):
        locked = plan.read(plan.ROOT / "models.lock.json")
        teacher, student = locked["models"]["teacher"], locked["models"]["public_student"]
        self.assertEqual(teacher["revision"], "8c1c8750df7cf0c35ca524efb4b02fbb44273856")
        self.assertEqual(student["revision"], "9d38a9844f1fce1309bfb150c03525413d7ac5bf")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            a = next(row for row in teacher["files"] if row["path"] == name)
            b = next(row for row in student["files"] if row["path"] == name)
            self.assertEqual(a, b)
        self.assertEqual(teacher["saved_config"], {"tie_word_embeddings": True, "scale_decoder_outputs": False})
        self.assertEqual(student["saved_transformers_version"], "5.20.2")
        for source in (teacher, student):
            self.assertEqual(len(next(row["sha256"] for row in source["files"]
                                      if row["path"] == "model.safetensors")), 64)

    def test_moving_revisions_unsafe_paths_and_invalid_hashes_are_rejected(self):
        scheme = plan.read(plan.ROOT / "scheme.json")
        original = plan.read(plan.ROOT / "models.lock.json")
        for kind in ("revision", "path", "hash", "size"):
            changed = copy.deepcopy(original)
            model = changed["models"]["teacher"]
            if kind == "revision":
                model["revision"] = "main"
            elif kind == "path":
                model["files"][0]["path"] = "../outside"
            elif kind == "hash":
                model["files"][0]["git_blob_sha1"] = "not a digest"
            else:
                model["files"][0]["bytes"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                plan.validate(scheme, changed)

    def test_plan_reads_only_metadata_without_execution(self):
        for name in ("torch", "transformers", "datasets", "numpy"):
            self.assertNotIn(name, sys.modules)
        existed = (plan.ROOT / ".venv-native").exists()
        result = plan.build()
        self.assertIn("validation", result["warning"])
        self.assertFalse(result["implementation_required"])
        self.assertEqual((plan.ROOT / ".venv-native").exists(), existed)


if __name__ == "__main__":
    unittest.main()
