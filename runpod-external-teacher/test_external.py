"""Tiny-file end-to-end provenance and controller tests. No real ML/GPU/network."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_teacher
from external_workflow import acquire, common, controller, pair
from external_workflow import models
from external_workflow import data as external_data, evaluation, smoke
from gridlib import inputs, policy
from sftlib import modeling
from sftlib.core import atomic_json, read_json, sha256, tree_manifest

RUNTIME = {"environment": {"python": "fake", "platform": "fake", "packages": {"transformers": "5.20.2"}},
           "gpu": {"name": "fake"}}


def artifact(root, relative, expected, files, **metadata):
    final = root / relative
    scratch = common.temporary(final)
    for name, value in files.items():
        path = common.owned(scratch, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, (dict, list)):
            atomic_json(path, value)
        else:
            path.write_bytes(value if isinstance(value, bytes) else value.encode())
    common.publish(scratch, final, expected, **metadata)
    return sha256(final / "COMPLETE.json")


def fixture(work, mode="public", full_rows=True, thresholds=None):
    root = work / "pair"
    root.mkdir(parents=True)
    locked = read_json(common.ROOT / "models.lock.json")
    payloads = {}
    for label, source in [*locked["models"].items(), ("xsum", locked["evaluation_data"])]:
        payloads[label] = {}
        for entry in source["files"]:
            payload = f"tiny {label} {entry['path']}".encode()
            entry["bytes"] = len(payload)
            if "sha256" in entry:
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
            else:
                entry["git_blob_sha1"] = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            payloads[label][entry["path"]] = payload
    atomic_json(root / "models.lock.json", locked)
    definition = {"student": mode, "models_lock_sha256": sha256(root / "models.lock.json"),
                  "restored_checkpoint_complete_sha256": "e" * 64 if mode == "restored" else None,
                  "thresholds": thresholds or {"teacher_r2": 20., "advantage": 1., "ci_lower": 0.},
                  "grid_work": str(work.parent / "grid")}
    atomic_json(work / "run.json", {"definition": definition})
    atomic_json(work / "native_runtime.json", RUNTIME)
    hashes = {}
    names = pair.expected_artifacts(definition)
    for label in (["teacher", "xsum", "public_student"] if mode == "public" else ["teacher", "xsum"]):
        spec = locked["evaluation_data"] if label == "xsum" else locked["models"][label]
        name = label + "_source"
        hashes[name] = artifact(root, names[name], pair.stage_identity(definition, "source", label=label, source=spec),
                                payloads[label])
    token_report = {"vocab_sha256": "a" * 64, "encoding_sha256": "b" * 64, "decoding_sha256": "c" * 64,
                    "vocabulary_entries": 32100, "max_token_id": 32099, "padded_output_rows": 28,
                    "counts": {"edge": 8, "validation": 11332, "train": 2048},
                    "special_ids": {"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2}}
    hashes["tokenizers"] = artifact(root, names["tokenizers"], pair.stage_identity(definition, "tokenizers"),
                                    {"report.json": {"student": token_report, "teacher": token_report}})
    hashes["corpus"] = artifact(root, names["corpus"], pair.stage_identity(definition, "corpus"), {"rows.jsonl": "{}\n"})
    inv = {"scale_decoder_outputs": False, "shapes": {"shared.weight": [1, 1]}, "unique_parameters": 1,
           "parameter_aliases": [["shared.weight", "encoder.embed_tokens.weight"]], "storage_aliases": []}
    parity = {"rtol": 1e-4, "atol": 1e-4, "max_abs_error": 0.}
    if mode == "restored":
        hashes["legacy_tokenizer"] = artifact(root, names["legacy_tokenizer"],
            pair.stage_identity(definition, "legacy_tokenizer"), {"report.json": {"student": token_report}})
        hashes["legacy_probe"] = artifact(root, names["legacy_probe"], pair.stage_identity(definition, "legacy_probe"),
            {"report.json": {"source_complete_sha256": "e" * 64, "inventory": inv}, "probe.safetensors": b"fake"})
        hashes["restored_source"] = artifact(root, names["restored_source"],
            pair.stage_identity(definition, "restored_source", checkpoint_complete_sha256="e" * 64), {"source.json": {}})
    n = 11332 if full_rows else 1
    ids, refs = [str(i) for i in range(n)], ["reference"] * n
    hashes["prepared"] = artifact(root, names["prepared"], pair.stage_identity(definition, "prepared"), {
        "data.json": {"counts": {"train": 204045, "validation": 11332}, "protocol": policy.PROTOCOL,
                      "ids_sha256": {"validation": common.digest(ids)},
                      "validation_references_sha256": common.digest(refs),
                      "source_complete_sha256": hashes["xsum_source"],
                      "tokenizer_report_sha256": hashes["tokenizers"]},
        "train/data.arrow": b"tiny", "validation/data.arrow": b"tiny"})
    for role in ("student", "teacher"):
        origin = (definition["restored_checkpoint_complete_sha256"] if role == "student" and mode == "restored"
                  else hashes["public_student_source" if role == "student" else "teacher_source"])
        hashes[role] = artifact(root, names[role], pair.stage_identity(definition, "model", role=role), {
            "config.json": {"external_pair_profile": common.PROFILE, "scale_decoder_outputs": False,
                            "vocab_size": 32128, "decoder_start_token_id": 0,
                            "pad_token_id": 0, "eos_token_id": 1},
            "model.safetensors": b"tiny fp weights",
            "provenance.json": {"local_sft_updates": 0, "profile": common.PROFILE, "runtime": RUNTIME,
                                "origin": "restored-local-sft" if role == "student" and mode == "restored"
                                          else "external-public-sft",
                                "historical_steps": 100000 if role == "student" else 8000,
                                "historical_steps_status": "verified-local-completion" if role == "student"
                                                          and mode == "restored" else "publisher-reported",
                                "source_complete_sha256": origin, "inventory": inv, "reload_parity": parity,
                                "migration": parity, "fp32_bf16_drift": {"mean_forward_kl": 0.001}}},
            role=role, precision="float32" if role == "student" else "bfloat16")
        score = 10. if role == "student" else 21.
        predictions = "".join(json.dumps({"id": id_, "reference": ref, "prediction": "summary",
            "scores": {key: score for key in ("rouge1", "rouge2", "rougeL")}}) + "\n" for id_, ref in zip(ids, refs))
        metrics = {"status": "complete", "n": 11332, "split": "validation", "precision": "bfloat16",
                   "scores": {key: score for key in ("rouge1", "rouge2", "rougeL")},
                   "counts": {"empty_prediction": 0, "hit_generation_cap": 0},
                   "runtime": RUNTIME, "predictions_sha256": hashlib.sha256(predictions.encode()).hexdigest()}
        hashes[role + "_validation"] = artifact(root, names[role + "_validation"],
            pair.stage_identity(definition, "validation", role=role, model_complete_sha256=hashes[role],
                                data_complete_sha256=hashes["prepared"], protocol=policy.PROTOCOL),
            {"metrics.json": metrics, "predictions.jsonl": predictions})
    hashes["quality"] = artifact(root, names["quality"], pair.stage_identity(definition, "quality",
        validation={role: hashes[role + "_validation"] for role in ("teacher", "student")}),
        {"report.json": {"passed": bool(21. >= definition["thresholds"]["teacher_r2"]
                                       and 11. >= definition["thresholds"]["advantage"]
                                       and 10. > definition["thresholds"]["ci_lower"]),
                         "n": 11332, "scores": {"teacher": 21., "student": 10.},
                         "advantage": 11., "paired_bootstrap_95ci": [10., 12.], "thresholds": definition["thresholds"]}})
    hashes["smoke"] = artifact(root, names["smoke"], pair.stage_identity(definition, "smoke",
        models={role: hashes[role] for role in ("student", "teacher")}), {"report.json": {"passed": True}})
    manifest = {"kind": common.PAIR_KIND, "status": "complete", "definition": definition, "local_sft_updates": 0,
        "protocol": policy.PROTOCOL, "runtime": RUNTIME, "models_lock_sha256": definition["models_lock_sha256"],
        "artifacts": {key: {"path": relative, "complete_sha256": hashes[key]} for key, relative in names.items()},
        "approval": {"accepted": True, "definition_sha256": common.digest(definition),
                     "quality_complete_sha256": hashes["quality"], "smoke_complete_sha256": hashes["smoke"]}}
    atomic_json(root / "prepared_pair.json", manifest)
    return root, manifest


class PairTests(unittest.TestCase):
    def test_code_changes_record_provenance_but_do_not_rebind_or_block_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root, _ = fixture(Path(d) / "work", full_rows=True)
            grid = Path(d) / "grid"
            with patch.object(pair, "environment", return_value=RUNTIME["environment"]), \
                    patch.object(inputs, "code_signature", return_value={"source": "old"}):
                inputs.binding(grid, root, policy.settings(), create=True)
            before = (grid / "binding.json").read_bytes()
            with patch.object(pair, "environment", return_value=RUNTIME["environment"]), \
                    patch.object(inputs, "code_signature", return_value={"source": "new"}):
                resumed = inputs.binding(grid, root, policy.settings(), full=True)
            self.assertEqual(resumed["code"], {"source": "new"})
            self.assertEqual((grid / "binding.json").read_bytes(), before)
            self.assertTrue(list((grid / "code_migrations").glob("*.json")))

    def test_honest_external_pair_and_grid_adapter_accept_without_sft_optimizer_state(self):
        for mode in ("public", "restored"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root, manifest = fixture(Path(d), mode)
                before = tree_manifest(root)
                with patch.object(pair, "environment", return_value=RUNTIME["environment"]):
                    result = inputs.inspect_pair(root, full=True)
                self.assertEqual(result["kind"], common.PAIR_KIND)
                self.assertEqual(result["paths"]["student_checkpoint"], str(root / "models/student_fp32"))
                self.assertFalse((root / "models/student_fp32/training_state.pt").exists())
                self.assertEqual(tree_manifest(root), before)

    def test_missing_approval_foreign_protocol_bad_paths_or_runtime_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root, original = fixture(Path(d), full_rows=False)
            for kind in ("approval", "protocol", "path", "mode", "runtime"):
                changed = copy.deepcopy(original)
                if kind == "approval":
                    changed["approval"]["accepted"] = False
                elif kind == "protocol":
                    changed["protocol"]["loss_temperature"] = 2.
                elif kind == "path":
                    changed["artifacts"]["teacher"]["path"] = "../outside"
                elif kind == "mode":
                    changed["definition"]["student"] = "automatic"
                else:
                    changed["runtime"]["environment"]["packages"]["transformers"] = "4.57.6"
                atomic_json(root / "prepared_pair.json", changed)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    pair.inspect(root, full=False, check_runtime=False)

    def test_incomplete_prediction_coverage_is_not_accepted_as_full_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root, _ = fixture(Path(d), full_rows=False)
            with self.assertRaisesRegex(ValueError, "IDs/references"):
                pair.inspect(root, full=True, check_runtime=False)

    def test_tampered_source_weights_or_unmanifested_weight_override_is_rejected(self):
        for kind in ("modified", "extra"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root, _ = fixture(Path(d), full_rows=False)
                if kind == "modified":
                    (root / "models/teacher_bf16/model.safetensors").write_bytes(b"corrupt")
                else:
                    (root / "models/teacher_bf16/model.safetensors.index.json").write_text("{}")
                with self.assertRaises(ValueError):
                    pair.inspect(root, full=kind == "modified", check_runtime=False)

    def test_existing_sft_pair_dispatch_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            atomic_json(root / "prepared_pair.json", {"status": "complete", "roles": {}})
            with self.assertRaisesRegex(ValueError, "Both SFT roles"):
                inputs.inspect_pair(root)


class DownloadTests(unittest.TestCase):
    def test_raw_sha256_and_git_blob_sha1_are_distinct_and_verified(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "file"
            value = b"hello\n"
            path.write_bytes(value)
            sha = hashlib.sha256(value).hexdigest()
            blob = hashlib.sha1(b"blob 6\0" + value).hexdigest()
            self.assertTrue(acquire.matches(path, {"bytes": 6, "sha256": sha}))
            self.assertTrue(acquire.matches(path, {"bytes": 6, "git_blob_sha1": blob}))
            self.assertFalse(acquire.matches(path, {"bytes": 6, "git_blob_sha1": hashlib.sha1(value).hexdigest()}))

    def test_transfer_resumes_only_own_partial_and_does_not_replace_corrupt_verified_source(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 3-5/6"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.safetensors"
            item = {"bytes": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest()}
            path.with_name(path.name + ".part").write_bytes(b"abc")
            with patch.object(acquire.urllib.request, "urlopen", return_value=Response(b"def")) as network, \
                    contextlib.redirect_stderr(io.StringIO()):
                acquire.transfer("https://example.invalid/file", path, item)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(network.call_args.args[0].get_header("Range"), "bytes=3-")
            path.write_bytes(b"broken")
            with patch.object(acquire.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "not overwritten"):
                    acquire.transfer("https://example.invalid/file", path, item)
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), b"broken")

    def test_path_escape_and_foreign_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                common.owned(root, "../outside")
            artifact(root, "sealed", {"owner": 1}, {"data": b"data"})
            with self.assertRaises(ValueError):
                common.verify(root / "sealed", {"owner": 2})


class ControllerTests(unittest.TestCase):
    def execute(self, root, accept_pair=False, accept_grid=False, quality=True, pause_grid=False, oom=False):
        work, grid = root / "work", root / "grid"
        args = SimpleNamespace(pod_disk=root, work_dir=work, grid_work=grid, student="public", restored_root=None,
            reference_python=None, min_teacher_r2=20., min_advantage=1., min_ci_lower=0.,
            minimum_free_gib=0., settings=common.REPO / "runpod-gkd/settings.json",
            accept_pair=accept_pair, accept_grid_estimate=accept_grid)
        calls, batches = [], []
        failed = [False]
        def worker(self, stage, python=None, batch=None):
            calls.append(stage)
            if stage.endswith("-validation"):
                batches.append((stage, batch))
                if oom and stage == "teacher-validation" and not failed[0]:
                    failed[0] = True
                    raise controller.ExternalOOM("typed")
            if stage == "quality":
                atomic_json(work / "pair/reports/quality/report.json", {"passed": quality, "scores": {}})
            return True
        def call(self, argv, name, **kwargs):
            calls.append(name)
            if name == "grid-pilot":
                atomic_json(grid / "status.json", {"status": "pilot_ready"})
            if name == "grid-run":
                atomic_json(grid / "status.json", {"status": "paused" if pause_grid else "complete"})
            self_args = argv
            if any("prepare_models.py" in arg for arg in self_args):
                raise AssertionError("External workflow must never launch SFT")
            return True
        with patch.object(controller, "operational_paths", return_value=(root, work, grid)), \
                patch.object(controller, "reviewed_pair", return_value={}), \
                patch.dict(os.environ, {}, clear=False), \
                patch.object(controller, "configure_local_storage"), patch.object(controller, "runtime"), \
                patch.object(controller, "check_space"), patch.object(controller.Runner, "worker", worker), \
                patch.object(controller.Runner, "call", call), patch.object(pair, "approve") as approval, \
                contextlib.redirect_stdout(io.StringIO()):
            if quality:
                controller.run(args)
            else:
                with self.assertRaisesRegex(ValueError, "failed quality"):
                    controller.run(args)
        return work, calls, batches, approval.call_count

    def test_default_pauses_before_pair_binding(self):
        with tempfile.TemporaryDirectory() as d:
            work, calls, _, approvals = self.execute(Path(d))
            self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_pair_approval")
            self.assertEqual(approvals, 0)
            self.assertNotIn("grid-bind", calls)
            self.assertEqual(calls[-1], "smoke")
            self.assertEqual(calls[:2], ["native-runtime", "acquire"])

    def test_pair_approval_runs_pilots_then_pauses_for_estimate(self):
        with tempfile.TemporaryDirectory() as d:
            work, calls, _, approvals = self.execute(Path(d), accept_pair=True)
            self.assertEqual(approvals, 1)
            self.assertEqual(calls[-3:], ["grid-bind", "grid-pilot", "grid-estimate"])
            self.assertEqual(read_json(work / "status.json")["phase"], "awaiting_grid_approval")

    def test_both_approvals_run_and_audit_full_grid_but_not_after_graceful_pause(self):
        for pause in (False, True):
            with self.subTest(pause=pause), tempfile.TemporaryDirectory() as d:
                work, calls, _, _ = self.execute(Path(d), accept_pair=True, accept_grid=True, pause_grid=pause)
                self.assertEqual("grid-audit" in calls, not pause)
                self.assertEqual(read_json(work / "status.json")["phase"], "paused" if pause else "complete")

    def test_failed_quality_cannot_be_overridden_by_approval_flags(self):
        with tempfile.TemporaryDirectory() as d:
            _, calls, _, approvals = self.execute(Path(d), accept_pair=True, accept_grid=True, quality=False)
            self.assertEqual(approvals, 0)
            self.assertNotIn("smoke", calls)
            self.assertFalse(any(name.startswith("grid-") for name in calls))

    def test_validation_oom_halves_batch_in_a_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            work, _, batches, _ = self.execute(Path(d), oom=True)
            self.assertEqual(batches, [("student-validation", 16), ("teacher-validation", 4), ("teacher-validation", 2)])
            self.assertEqual(read_json(work / "teacher_validation_batch.json")["batch_size"], 2)

    def test_worker_command_preserves_virtualenv_symlink_path(self):
        python = Path("/workspace/code/.venv-reference/bin/python")
        argv = controller.worker_command(Path("/workspace/work"), "legacy-probe", python)
        self.assertEqual(argv[0], str(python))
        self.assertIn("-B", argv)
        self.assertNotIn("train", argv)

    def test_plan_imports_no_ml_or_network_and_writes_no_work_directory(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "absent"
            with patch("sys.argv", ["external_teacher.py", "--work-dir", str(work), "--student", "public", "plan"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                external_teacher.main()
            self.assertFalse(work.exists())
            for name in ("torch", "transformers", "datasets", "numpy"):
                self.assertNotIn(name, sys.modules)


class ModelContractTests(unittest.TestCase):
    def test_fp32_probe_generation_does_not_silently_use_bf16_autocast(self):
        class IDs:
            def __getitem__(self, index):
                return self
            def eq(self, value):
                return self
            def all(self):
                return True
        model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0),
                                generate=lambda **kwargs: IDs())
        tok = SimpleNamespace(eos_token_id=1, pad_token_id=0)
        def generation_config(**kwargs):
            self.assertNotIn("min_length", kwargs)
            self.assertEqual(kwargs["min_new_tokens"], 0)
            return SimpleNamespace(**kwargs)
        torch = SimpleNamespace(no_grad=contextlib.nullcontext, bfloat16="bf16",
                                autocast=lambda *a, **k: contextlib.nullcontext())
        with patch.dict("sys.modules", {"torch": torch,
                "transformers": SimpleNamespace(GenerationConfig=generation_config)}), \
                patch.object(torch, "autocast", wraps=torch.autocast) as autocast:
            models.greedy(model, tok, {}, bf16=False)
            autocast.assert_not_called()
            models.greedy(model, tok, {}, bf16=True)
            autocast.assert_called_once_with("cuda", dtype="bf16")

    def test_native_grid_loader_enforces_version_scaling_eager_and_uniform_bf16(self):
        class Model:
            config = SimpleNamespace(use_cache=True)
            def to(self, dtype):
                self.converted = dtype
                return self
            def requires_grad_(self, value):
                self.trainable = value
                return self
            def eval(self):
                self.evaluating = True
                return self
        cfg = SimpleNamespace(external_pair_profile=common.PROFILE, scale_decoder_outputs=False,
                              model_type="t5", is_encoder_decoder=True, vocab_size=32128,
                              decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        torch = SimpleNamespace(float32="fp32", bfloat16="bf16")
        factory = SimpleNamespace(from_pretrained=lambda *a, **k: Model())
        transformers = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: cfg),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: "tokenizer"),
            AutoModelForSeq2SeqLM=factory)
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}), \
                patch("importlib.metadata.version", return_value="5.20.2"), \
                patch.object(factory, "from_pretrained", wraps=factory.from_pretrained) as load:
            _, model = modeling.load(Path("artifact"), "cuda", False, {"dropout": 0})
            self.assertEqual(load.call_args.kwargs["dtype"], "bf16")
            self.assertNotIn("torch_dtype", load.call_args.kwargs)
            self.assertEqual(load.call_args.kwargs["attn_implementation"], "eager")
            self.assertEqual(model.converted, "bf16")
            self.assertFalse(model.trainable)
            cfg.scale_decoder_outputs = True
            with self.assertRaisesRegex(ValueError, "scaling"):
                modeling.load(Path("artifact"), "cuda", False, {})
            cfg.scale_decoder_outputs = False
            with patch("importlib.metadata.version", return_value="4.57.6"):
                with self.assertRaisesRegex(ValueError, "isolated native"):
                    modeling.load(Path("artifact"), "cuda", False, {})

    def test_parameter_alias_inventory_tracks_objects_not_just_config_flags(self):
        class Parameter:
            shape = (2, 3)
            dtype = "float32"
            def numel(self):
                return 6
            def storage_offset(self):
                return 0
            def untyped_storage(self):
                return SimpleNamespace(data_ptr=lambda: id(self))
        shared, head = Parameter(), Parameter()
        def names(remove_duplicate=True):
            return ([("shared.weight", shared), ("lm_head.weight", head)] if remove_duplicate else
                    [("shared.weight", shared), ("encoder.embed_tokens.weight", shared), ("lm_head.weight", head)])
        model = SimpleNamespace(config=SimpleNamespace(tie_word_embeddings=True, scale_decoder_outputs=False),
                                named_parameters=names, named_buffers=lambda: [])
        inventory = models.inventory(model)
        self.assertFalse(inventory["scale_decoder_outputs"])
        self.assertEqual(inventory["unique_parameters"], 12)
        self.assertEqual(inventory["parameter_aliases"], [["encoder.embed_tokens.weight", "shared.weight"]])
        self.assertNotIn("lm_head.weight", inventory["parameter_aliases"][0])

    def test_native_source_loader_refuses_missing_weights_instead_of_reinitializing(self):
        cfg = SimpleNamespace(model_type="t5", is_encoder_decoder=True)
        transformers = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: cfg),
            AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=lambda *a, **k: (object(), {"missing_keys": ["lm_head.weight"]})))
        with patch.dict("sys.modules", {"torch": SimpleNamespace(float32="fp32", bfloat16="bf16"),
                                       "transformers": transformers}):
            with self.assertRaisesRegex(ValueError, "preserve all tensors"):
                models.load_model(Path("public"), "cuda")

    def test_public_student_export_has_real_file_hashes_and_no_fake_sft_state(self):
        class Model:
            config = SimpleNamespace(use_cache=False)
            def parameters(self):
                return [SimpleNamespace(numel=lambda: 1, element_size=lambda: 4)]
            def save_pretrained(self, path, **kwargs):
                atomic_json(path / "config.json", {"external_pair_profile": self.config.external_pair_profile,
                                                  "scale_decoder_outputs": False})
                (path / "model.safetensors").write_bytes(b"tiny weights")
        class IDs:
            def cpu(self):
                return self
        ids = IDs()
        tensors = {"generated": ids}
        inv = {"parameter_aliases": [], "storage_aliases": [], "unique_parameters": 1,
               "shapes": {"head": [1]}, "scale_decoder_outputs": False}
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            atomic_json(work / "run.json", {"definition": {"student": "public"}})
            source = work / "pair/sources/public_student"
            atomic_json(source / "COMPLETE.json", {"source": "fake locked source"})
            before = tree_manifest(source)
            tokenizer = SimpleNamespace(save_pretrained=lambda p: (p / "tokenizer.json").write_text("{}"))
            fake_safetensors = SimpleNamespace(load_file=lambda p: {}, save_file=lambda x, p: p.write_bytes(b"probe"))
            with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None),
                                                                   equal=lambda a, b: a is b),
                                           "safetensors": SimpleNamespace(torch=fake_safetensors),
                                           "safetensors.torch": fake_safetensors}), \
                    patch.object(models, "runtime", return_value=("cuda", RUNTIME)), \
                    patch.object(models, "verified_model_source", return_value=source), \
                    patch.object(models, "load_model", side_effect=[Model(), Model()]) as load, \
                    patch.object(models, "load_tokenizer", return_value=tokenizer), \
                    patch.object(models, "probe_rows", return_value=[]), \
                    patch.object(models, "inventory", return_value=inv), \
                    patch.object(models, "tensors_and_gradients", return_value=tensors), \
                    patch.object(models, "compare", return_value={"rtol": 1e-4, "atol": 1e-4, "max_abs_error": 0}), \
                    patch.object(models, "make_inputs", return_value=({}, None)), \
                    patch.object(models, "greedy", return_value=ids), patch.object(models, "check_space"):
                models.export(work, "student")
            self.assertEqual(load.call_count, 2)
            exported = work / "pair/models/student_fp32"
            common.verify(exported, common.identity(work, "model", role="student"))
            provenance = read_json(exported / "provenance.json")
            self.assertEqual(provenance["origin"], "external-public-sft")
            self.assertEqual(provenance["local_sft_updates"], 0)
            self.assertEqual(provenance["historical_steps_status"], "publisher-reported")
            self.assertFalse((exported / "training_state.pt").exists())
            self.assertEqual(tree_manifest(source), before)


class OOMHandshakeTests(unittest.TestCase):
    def test_only_typed_fresh_correct_attempt_report_causes_validation_retry(self):
        for kind in ("valid", "missing", "wrong_stage"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                work = Path(d)
                atomic_json(work / "run.json", {"definition": {"mode": "test"}})
                attempt = work / "attempts/one.json"
                def logged(argv, log, stop, **kwargs):
                    if kind != "missing":
                        atomic_json(attempt, {"status": "cuda_oom", "stage": (
                            "wrong" if kind == "wrong_stage" else "teacher-validation"),
                            "batch_size": 4, "definition_sha256": common.digest({"mode": "test"})})
                    return common.OOM_EXIT
                runner = controller.Runner(work, SimpleNamespace(requested=False))
                with patch.object(controller, "run_logged", side_effect=logged):
                    error_type = controller.ExternalOOM if kind == "valid" else RuntimeError
                    with self.assertRaises(error_type) as error:
                        runner.call(["worker"], "validation", oom_stage="teacher-validation", batch=4, attempt=attempt)
                    if kind != "valid":
                        self.assertNotIsInstance(error.exception, controller.ExternalOOM)


class EvidenceTests(unittest.TestCase):
    def test_one_disk_paths_are_read_only_and_reject_overlap_or_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            disk = Path(d).resolve()
            repo = disk / "code"
            code = repo / "runpod-external-teacher"
            for directory in (code, repo / "runpod-sft", repo / "runpod-gkd"):
                directory.mkdir(parents=True, exist_ok=True)
            with patch.object(common, "ROOT", code), patch.object(common, "REPO", repo):
                self.assertEqual(common.operational_paths(disk, disk / "work", disk / "grid")[:2],
                                 (disk, disk / "work"))
                self.assertFalse((disk / "work").exists())
                self.assertFalse((disk / "grid").exists())
                for work, grid, restored in (
                    (repo / "work", disk / "grid", None),
                    (disk / "work", disk / "work/grid", None),
                    (disk / "results_and_temp/work", disk / "grid", None),
                    (disk / "work", disk / "grid", disk / "work/student"),
                ):
                    with self.subTest(work=work, grid=grid), self.assertRaises(ValueError):
                        common.operational_paths(disk, work, grid, restored)

    def test_raw_official_data_rejects_duplicates_and_cross_split_overlap(self):
        class Table(list):
            def __getitem__(self, key):
                return [row[key] for row in self] if isinstance(key, str) else super().__getitem__(key)
        for kind in ("valid", "duplicates", "overlap"):
            train = Table([{"id": "a"}, {"id": "a" if kind == "duplicates" else "b"}])
            validation = Table([{"id": "a" if kind == "overlap" else "c"}])
            with self.subTest(kind=kind), patch.dict("sys.modules", {
                    "datasets": SimpleNamespace(load_dataset=lambda *a, **k: {"train": train, "validation": validation})}), \
                    patch.object(external_data, "verify"), \
                    patch.object(external_data, "COUNTS", {"train": 2, "validation": 1}):
                if kind == "valid":
                    self.assertEqual(external_data.raw(Path("unused"))["train"], train)
                else:
                    with self.assertRaises(ValueError):
                        external_data.raw(Path("unused"))

    def test_approval_cannot_be_given_before_reports_exist(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "without --accept-pair"):
                controller.reviewed_pair(Path(d))

    def test_approval_requires_passed_sealed_reports_and_binds_their_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            for stage in ("quality", "smoke"):
                artifact(work / "pair", "reports/" + stage, {"stage": stage},
                         {"report.json": {"passed": True}})
            reviewed = controller.reviewed_pair(work)
            self.assertEqual(set(reviewed), {"quality", "smoke"})
            (work / "pair/reports/smoke/report.json").write_text('{"passed":false}')
            with self.assertRaises(ValueError):
                controller.reviewed_pair(work)

    def test_invalid_pair_is_not_published_even_with_a_passed_quality_flag(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            root, _ = fixture(work, full_rows=False)
            (root / "prepared_pair.json").unlink()  # Only a fixture produced by this test.
            with patch.object(pair, "environment", return_value=RUNTIME["environment"]):
                with self.assertRaisesRegex(ValueError, "IDs/references"):
                    pair.approve(work)
            self.assertFalse((root / "prepared_pair.json").exists())

    def test_valid_pair_publishes_and_cached_approval_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            root, _ = fixture(work)
            (root / "prepared_pair.json").unlink()
            with patch.object(pair, "environment", return_value=RUNTIME["environment"]):
                pair.approve(work)
                before = tree_manifest(root)
                pair.approve(work)
            self.assertEqual(tree_manifest(root), before)

    def test_native_tokenizer_mismatch_stops_before_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            atomic_json(work / "run.json", {"definition": {"student": "public"}})
            a = {"vocab_sha256": "a", "encoding_sha256": "x", "decoding_sha256": "x",
                 "special_ids": {}, "counts": {}}
            b = {**a, "encoding_sha256": "different"}
            with patch.object(external_data, "runtime"), \
                    patch.object(external_data, "audit_tokenizer", side_effect=[a, b]):
                with self.assertRaisesRegex(ValueError, "No automatic swap"):
                    external_data.token_audit(work)
            self.assertFalse((work / "pair/reports/tokenizers").exists())

    def test_wrong_runtime_is_rejected_even_when_all_reports_are_valid(self):
        with tempfile.TemporaryDirectory() as d:
            root, _ = fixture(Path(d), full_rows=False)
            with patch.object(pair, "environment", return_value={"packages": {"transformers": "4.57.6"}}):
                with self.assertRaisesRegex(ValueError, "exact validated native"):
                    pair.inspect(root, full=False)

    def test_artifact_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "outside").write_text("user data")
            try:
                (root / "link").symlink_to(root / "outside")
            except OSError:
                self.skipTest("Creating symlinks requires privileges on this platform")
            with self.assertRaises(ValueError):
                common.owned(root, "link")
            self.assertEqual((root / "outside").read_text(), "user data")


class EvaluationTests(unittest.TestCase):
    def test_worker_uses_bf16_full_reference_coverage_and_publishes_only_after_success(self):
        class Table(list):
            def __getitem__(self, index):
                return [row[index] for row in self] if isinstance(index, str) else super().__getitem__(index)
        class Generated:
            def __init__(self, rows):
                self.rows = rows
            def cpu(self):
                return self
            def tolist(self):
                return [[7, 1] for _ in self.rows]
        rows = Table({"id": str(i), "input_ids": [5, 1], "labels": [7, 1],
                      "reference": "entire human reference " * 80,
                      "source_truncated": False, "reference_over_cap": True} for i in range(3))
        tok = SimpleNamespace(eos_token_id=1, decode=lambda tokens, **kwargs: "summary")
        scorer = SimpleNamespace(score=lambda ref, text: {
            key: SimpleNamespace(fmeasure=0.5) for key in ("rouge1", "rouge2", "rougeL")})
        fake_rouge = SimpleNamespace(rouge_scorer=SimpleNamespace(RougeScorer=lambda *a, **k: scorer))
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            atomic_json(work / "run.json", {"definition": {"student": "public"}})
            atomic_json(work / "pair/models/student_fp32/COMPLETE.json", {"tiny": "model"})
            atomic_json(work / "pair/student/prepared/COMPLETE.json", {"tiny": "data"})
            with patch.dict("sys.modules", {"torch": SimpleNamespace(
                    cuda=SimpleNamespace(synchronize=lambda device: None)), "rouge_score": fake_rouge}), \
                    patch.object(evaluation, "VALIDATION_N", 3), patch.object(evaluation, "verify"), \
                    patch.object(evaluation, "runtime", return_value=("cuda", RUNTIME)), \
                    patch.object(evaluation, "load_model", return_value=object()) as model, \
                    patch.object(evaluation, "load_tokenizer", return_value=tok), \
                    patch.object(evaluation, "load_data", return_value={"validation": rows}), \
                    patch("sftlib.data.batch", side_effect=lambda tok, rows, device: (rows, None)), \
                    patch("sftlib.modeling.generate", side_effect=lambda model, tok, rows, cap: Generated(rows)):
                evaluation.evaluate(work, "student", 2)
                self.assertEqual(model.call_args.kwargs["precision"], "bfloat16")
            final = work / "pair/validation/student"
            common.verify(final)
            predictions = [json.loads(line) for line in (final / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in predictions], ["0", "1", "2"])
            self.assertTrue(all(row["reference"] == rows[0]["reference"] for row in predictions))
            self.assertTrue(all(row["generated_ids"] == [7, 1] for row in predictions))
            metrics = read_json(final / "metrics.json")
            self.assertEqual(metrics["scores"]["rouge2"], 50.)
            self.assertEqual(metrics["rates"]["reference_over_cap"], 1.)

    def test_quality_uses_paired_per_example_differences_and_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            root, manifest = fixture(work)
            # Delete only this test's generated quality artifact before exercising its producer.
            import shutil
            shutil.rmtree(root / "reports/quality")
            with patch.object(evaluation, "bootstrap", return_value=[10., 12.]) as bootstrap:
                evaluation.quality(work)
            diffs = bootstrap.call_args.args[0]
            self.assertEqual(len(diffs), 11332)
            self.assertEqual(set(diffs), {11.})
            report = read_json(root / "reports/quality/report.json")
            self.assertTrue(report["passed"])
            self.assertEqual(report["thresholds"], manifest["definition"]["thresholds"])


class SmokeTests(unittest.TestCase):
    def test_real_engine_handoff_checks_restart_and_discards_only_owned_smoke_models(self):
        from gridlib import backend, engine
        from sftlib import training
        tensor_type = type("FakeTensor", (), {})
        fake_torch = SimpleNamespace(Tensor=tensor_type, cuda=SimpleNamespace(empty_cache=lambda: None),
                                     load=lambda path, **kwargs: read_json(path))
        parameter = SimpleNamespace(_version=0, requires_grad=False, grad=None)
        teacher = SimpleNamespace(training=False, parameters=lambda: [parameter])
        calls = []
        def train(root, bound, item, selected, session_updates, microbatch):
            calls.append((root.name, microbatch, bound["inputs"]["paths"]["student_checkpoint"]))
            run_dir = root / "cells" / item["id"]
            latest = run_dir / "latest.json"
            step = read_json(latest)["step"] + 1 if latest.exists() else 1
            directory = run_dir / "checkpoints" / f"step-{step:08d}"
            atomic_json(run_dir / "run.json", {"run_id": "disposable"})
            atomic_json(directory / "COMPLETE.json", {"step": step, "run_id": "disposable"})
            state = {**engine.new_state(), "step": step, "cursor": 32 * step, "points": 48 * step,
                     "on_policy_examples": 16 * step, "off_policy_examples": 16 * step,
                     "training_seconds": 1. if root.name == "restart_a" else 2.}
            atomic_json(directory / "training_state.pt", {"step": step, "accounting": state,
                "optimizer": {"step": step}, "rng": step, "branch_rng": step})
            atomic_json(latest, {"step": step})
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            atomic_json(work / "run.json", {"definition": {"student": "public"}})
            for role in ("student_fp32", "teacher_bf16"):
                atomic_json(work / "pair/models" / role / "COMPLETE.json", {"tiny": role})
            immutable = tree_manifest(work / "pair/models")
            with patch.dict("sys.modules", {"torch": fake_torch}), \
                    patch.object(smoke, "runtime"), patch.object(smoke, "verify"), \
                    patch.object(smoke, "weights", return_value={"weight": 1}), \
                    patch.object(backend, "load_pair", return_value=(
                        "cuda", {"runtime": "fake"}, object(), object(), teacher, {"train": list(range(32))})), \
                    patch.object(backend, "optimizer"), patch.object(backend, "update", return_value={"loss": .1}), \
                    patch.object(training, "seed"), patch.object(engine, "train", side_effect=train):
                smoke.run(work)
            final = work / "pair/reports/smoke"
            common.verify(final)
            report = read_json(final / "report.json")
            self.assertTrue(report["passed"])
            self.assertEqual([mb for _, mb, _ in calls], [2, 2, 2, 1])
            self.assertEqual(len({path for _, _, path in calls}), 1)
            self.assertFalse((final / "restart_a").exists())
            self.assertFalse((final / "restart_b").exists())
            self.assertEqual(tree_manifest(work / "pair/models"), immutable)


if __name__ == "__main__":
    unittest.main()
