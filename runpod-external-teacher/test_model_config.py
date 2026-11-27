"""Untied derived-config normalization; no real ML, GPU or model downloads."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_workflow import common, models
from sftlib.core import atomic_json, tree_manifest


class Parameter:
    shape = (2, 3)
    dtype = "float32"

    def __init__(self, values, storage=None):
        self.values = tuple(values)
        self.storage = id(self) if storage is None else storage
        self.grad = object()

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage)

    def storage_offset(self):
        return 0

    def numel(self):
        return 6


class Model:
    def __init__(self, tied=False, equal_values=False):
        self.config = SimpleNamespace(tie_word_embeddings=True, scale_decoder_outputs=False)
        self.shared = Parameter(range(6))
        self.head = self.shared if tied else Parameter(range(6) if equal_values else range(6, 12))

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.shared)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.head)

    def named_parameters(self, remove_duplicate=True):
        pairs = [("shared.weight", self.shared), ("encoder.embed_tokens.weight", self.shared),
                 ("decoder.embed_tokens.weight", self.shared), ("lm_head.weight", self.head)]
        if not remove_duplicate:
            return pairs
        found, unique = set(), []
        for name, parameter in pairs:
            if id(parameter) not in found:
                found.add(id(parameter))
                unique.append((name, parameter))
        return unique

    def named_buffers(self):
        return []

    def tie_weights(self):
        raise AssertionError("Normalization must never tie or replace parameters")


class ConfigTests(unittest.TestCase):
    def test_separate_parameters_normalize_without_changing_weights_gradients_or_aliases(self):
        model = Model()
        before = models.inventory(model)
        parameters = [id(model.shared), id(model.head)]
        gradients = [model.shared.grad, model.head.grad]
        values = [model.shared.values, model.head.values]
        result = models.normalize_export_config(model)
        self.assertFalse(model.config.tie_word_embeddings)
        self.assertFalse(model.config.scale_decoder_outputs)
        self.assertEqual(models.inventory(model), before)
        self.assertEqual([id(model.shared), id(model.head)], parameters)
        self.assertEqual([model.shared.grad, model.head.grad], gradients)
        self.assertEqual([model.shared.values, model.head.values], values)
        self.assertTrue(result["source_loaded_config"]["tie_word_embeddings"])
        self.assertFalse(result["export_config"]["tie_word_embeddings"])
        self.assertFalse(result["input_output_parameter_tied"])
        # Encoder/decoder INPUT sharing is not disabled by untied output metadata.
        self.assertIn(["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "shared.weight"],
                      before["parameter_aliases"])

    def test_equal_values_are_not_mistaken_for_shared_parameters(self):
        model = Model(equal_values=True)
        models.normalize_export_config(model)
        self.assertEqual(model.shared.values, model.head.values)
        self.assertIsNot(model.shared, model.head)
        self.assertFalse(model.config.tie_word_embeddings)

    def test_actually_tied_parameters_are_not_forced_untied(self):
        model = Model(tied=True)
        before = models.inventory(model)
        result = models.normalize_export_config(model)
        self.assertTrue(model.config.tie_word_embeddings)
        self.assertFalse(model.config.scale_decoder_outputs)
        self.assertTrue(result["input_output_parameter_tied"])
        self.assertEqual(models.inventory(model), before)

    def test_scaling_changes_and_ambiguous_storage_are_rejected_before_config_mutation(self):
        for kind in ("scaled", "shared-storage"):
            with self.subTest(kind=kind):
                model = Model()
                if kind == "scaled":
                    model.config.scale_decoder_outputs = True
                else:
                    model.head.storage = model.shared.storage
                before = copy.deepcopy(vars(model.config))
                with self.assertRaises(ValueError):
                    models.normalize_export_config(model)
                self.assertEqual(vars(model.config), before)
        model = Model()
        model.config.tie_word_embeddings = False
        del model.config.scale_decoder_outputs
        models.normalize_export_config(model)
        self.assertIs(model.config.scale_decoder_outputs, False)

    def test_completed_exports_are_reused_without_rewriting_their_configs(self):
        fake_safetensors = SimpleNamespace(load_file=lambda *a: None, save_file=lambda *a: None)
        for role, suffix in (("student", "fp32"), ("teacher", "bf16")):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as d:
                work = Path(d)
                atomic_json(work / "run.json", {"definition": {"student": "public"}})
                final = work / "pair/models" / f"{role}_{suffix}"
                scratch = common.temporary(final)
                atomic_json(scratch / "config.json", {"tie_word_embeddings": True, "scale_decoder_outputs": False})
                common.publish(scratch, final, common.identity(work, "model", role=role))
                before = tree_manifest(work)
                with patch.dict(sys.modules, {
                        "torch": SimpleNamespace(), "safetensors": SimpleNamespace(torch=fake_safetensors),
                        "safetensors.torch": fake_safetensors}), \
                        patch.object(models, "runtime", side_effect=AssertionError("No new GPU work")), \
                        patch.object(models, "normalize_export_config",
                                     side_effect=AssertionError("Do not normalize cached artifacts")):
                    models.export(work, role)
                self.assertEqual(tree_manifest(work), before)


if __name__ == "__main__":
    unittest.main()
