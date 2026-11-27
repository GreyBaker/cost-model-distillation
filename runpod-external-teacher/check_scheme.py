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

    def test_grid_totals_match_original_grid(self):
        totals = plan.build()["grid_totals"]
        self.assertEqual((totals["cells"], totals["points"], totals["examples"],
                          totals["updates"], totals["unspent_points"]), (25, 332471, 235266, 7363, 29))
        # Cross-check actual existing engine policy, not just copied constants.
        sys.path.insert(0, str(plan.ROOT.parent / "runpod-gkd"))
        from gridlib.policy import cells
        for expected, proposed in zip(cells(), plan.build()["grid_cells"]):
            self.assertTrue(all(expected[key] == value for key, value in proposed.items()))

    def test_all_approvals_precede_grid_training(self):
        stages = plan.build()["stages"]
        self.assertLess(stages.index("full_11332_teacher_and_student_validation_and_paired_difference"),
                        stages.index("publish_honest_external_pair_manifest_atomically"))
        self.assertLess(stages.index("review_and_explicitly_approve_quality_and_compatibility_reports"),
                        stages.index("publish_honest_external_pair_manifest_atomically"))
        self.assertNotIn("disposable_gkd_update_save_reload_resume_smoke_test", stages)
        self.assertEqual(plan.build()["development"], {
            "smoke_command": "external_teacher.py dev-smoke", "included_in_standard_run": False})
        self.assertLess(stages.index("print_measured_estimate_and_wait_for_grid_approval"),
                        stages.index("run_all_25_cells_independently_from_fixed_fp32_student"))

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
