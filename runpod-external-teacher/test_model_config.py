"""Source-config preservation and visible loading diagnostics; no real ML/GPU."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_workflow import common, models
from sftlib import modeling
from sftlib.core import atomic_json, tree_manifest


class Parameter:
    shape = (2, 3)
    dtype = "float32"

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self))

    def storage_offset(self):
        return 0

    def numel(self):
        return 6


class Model:
    def __init__(self, config):
        self.config = config
        self.shared, self.head = Parameter(), Parameter()

    def named_parameters(self, remove_duplicate=True):
        return [("shared.weight", self.shared), ("lm_head.weight", self.head)]

    def named_buffers(self):
        return []

    def eval(self):
        return self

    def train(self):
        return self


class ConfigTests(unittest.TestCase):
    def test_config_observation_is_read_only_and_normalization_is_removed(self):
        config = SimpleNamespace(tie_word_embeddings=True, scale_decoder_outputs=False)
        model = Model(config)
        before = models.inventory(model)
        self.assertFalse(hasattr(models, "normalize_export_config"))
        self.assertEqual(models.config_observation(model),
                         {"tie_word_embeddings": True, "scale_decoder_outputs": False})
        self.assertIsNot(model.shared, model.head)
        self.assertEqual(models.inventory(model), before)
        self.assertEqual(vars(config), {"tie_word_embeddings": True, "scale_decoder_outputs": False})

    def test_import_and_grid_loaders_leave_warnings_and_original_config_visible(self):
        for loader in ("import", "grid"):
            with self.subTest(loader=loader):
                config = SimpleNamespace(tie_word_embeddings=True, scale_decoder_outputs=False,
                    model_type="t5", is_encoder_decoder=True, vocab_size=32128,
                    decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
                model = Model(config)
                def load(*args, **kwargs):
                    self.assertIs(kwargs["config"], config)
                    self.assertTrue(config.tie_word_embeddings)
                    warnings.warn("distinct shared.weight/lm_head.weight", UserWarning)
                    logging.getLogger("transformers.modeling_utils").warning("untied checkpoint advisory")
                    return (model, {}) if loader == "import" else model
                transformers = SimpleNamespace(
                    AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
                    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
                    AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=load))
                with patch.dict(sys.modules, {
                        "torch": SimpleNamespace(float32="fp32", bfloat16="bf16"),
                        "transformers": transformers}), warnings.catch_warnings(record=True) as caught, \
                        self.assertLogs("transformers.modeling_utils", level="WARNING") as logs:
                    warnings.simplefilter("always")
                    if loader == "import":
                        self.assertIs(models.load_model(Path("source"), "cuda"), model)
                    else:
                        modeling.load(Path("source"), "cuda", True,
                                      {"dropout": 0., "gradient_checkpointing": False})
                self.assertTrue(any("distinct shared.weight" in str(w.message) for w in caught))
                self.assertIn("untied checkpoint advisory", logs.output[0])
                self.assertTrue(config.tie_word_embeddings)
                self.assertFalse(config.scale_decoder_outputs)

    def test_completed_exports_are_reused_without_rewriting_even_older_normalized_configs(self):
        fake_safetensors = SimpleNamespace(load_file=lambda *a: None, save_file=lambda *a: None)
        for role, suffix in (("student", "fp32"), ("teacher", "bf16")):
            for tied_flag in (True, False):
                with self.subTest(role=role, tied_flag=tied_flag), tempfile.TemporaryDirectory() as d:
                    work = Path(d)
                    atomic_json(work / "run.json", {"definition": {"student": "public"}})
                    final = work / "pair/models" / f"{role}_{suffix}"
                    scratch = common.temporary(final)
                    atomic_json(scratch / "config.json", {
                        "tie_word_embeddings": tied_flag, "scale_decoder_outputs": False})
                    common.publish(scratch, final, common.identity(work, "model", role=role))
                    before = tree_manifest(work)
                    with patch.dict(sys.modules, {
                            "torch": SimpleNamespace(), "safetensors": SimpleNamespace(torch=fake_safetensors),
                            "safetensors.torch": fake_safetensors}), \
                            patch.object(models, "runtime", side_effect=AssertionError("No new GPU work")):
                        models.export(work, role)
                    self.assertEqual(tree_manifest(work), before)


if __name__ == "__main__":
    unittest.main()
