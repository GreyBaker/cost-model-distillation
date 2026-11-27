"""Public-pair tiny files for focused waiver checks; no legacy test imports or ML."""
from __future__ import annotations

import hashlib
import json

from external_workflow import common, pair
from gridlib.policy import PROTOCOL, VALIDATION_N
from sftlib.core import atomic_json, read_json, sha256, tree_manifest

RUNTIME = {"environment": {"python": "fake", "platform": "fake", "packages": {"transformers": "5.20.2"}},
           "gpu": {"name": "fake"}}


def artifact(root, relative, identity, files, **metadata):
    """Synthetic sealed files; production publication is covered by the full suite."""
    directory = root / relative
    directory.mkdir(parents=True)
    for name, value in files.items():
        path = common.owned(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, (dict, list)):
            atomic_json(path, value)
        else:
            path.write_bytes(value if isinstance(value, bytes) else value.encode())
    atomic_json(directory / "COMPLETE.json", {
        "kind": common.ARTIFACT_KIND, "status": "complete", "identity": identity,
        "files": tree_manifest(directory), **metadata})
    return sha256(directory / "COMPLETE.json")


def failed_pair(work, *, metric_offsets=None, teacher_minimum=25.):
    """Complete evaluation, failed by default; optional offsets model float roundoff."""
    metric_offsets = metric_offsets or {}
    root = work / "pair"
    root.mkdir(parents=True)
    locked = read_json(common.ROOT / "models.lock.json")
    payloads = {}
    for label, source in [*locked["models"].items(), ("xsum", locked["evaluation_data"])]:
        payloads[label] = {}
        for entry in source["files"]:
            payload = f"tiny {label} {entry['path']}".encode()
            entry["bytes"] = len(payload)
            key = "sha256" if "sha256" in entry else "git_blob_sha1"
            entry[key] = (hashlib.sha256(payload).hexdigest() if key == "sha256" else
                          hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest())
            payloads[label][entry["path"]] = payload
    atomic_json(root / "models.lock.json", locked)
    definition = {"student": "public", "models_lock_sha256": sha256(root / "models.lock.json"),
                  "restored_checkpoint_complete_sha256": None,
                   "thresholds": {"teacher_r2": teacher_minimum, "advantage": 1., "ci_lower": 0.},
                  "grid_work": str(work.parent / "grid")}
    atomic_json(work / "run.json", {"definition": definition})
    atomic_json(work / "native_runtime.json", RUNTIME)
    hashes, names = {}, pair.expected_artifacts(definition)

    def seal(name, stage, files, *, metadata=None, **identity):
        hashes[name] = artifact(root, names[name], pair.stage_identity(definition, stage, **identity),
                                files, **(metadata or {}))

    for label in ("teacher", "xsum", "public_student"):
        spec = locked["evaluation_data"] if label == "xsum" else locked["models"][label]
        seal(label + "_source", "source", payloads[label], label=label, source=spec)
    tokens = {"vocab_sha256": "a" * 64, "encoding_sha256": "b" * 64, "decoding_sha256": "c" * 64,
              "vocabulary_entries": 32100, "max_token_id": 32099, "padded_output_rows": 28,
              "counts": {"edge": 8, "validation": VALIDATION_N, "train": 2048},
              "special_ids": {"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2}}
    seal("tokenizers", "tokenizers", {"report.json": {"student": tokens, "teacher": tokens}})
    seal("corpus", "corpus", {"rows.jsonl": "{}\n"})
    ids, refs = [str(i) for i in range(VALIDATION_N)], ["reference"] * VALIDATION_N
    seal("prepared", "prepared", {
        "data.json": {"counts": {"train": 204045, "validation": VALIDATION_N}, "protocol": PROTOCOL,
                      "ids_sha256": {"validation": common.digest(ids)},
                      "validation_references_sha256": common.digest(refs),
                      "source_complete_sha256": hashes["xsum_source"],
                      "tokenizer_report_sha256": hashes["tokenizers"]},
        "train/data.arrow": b"tiny", "validation/data.arrow": b"tiny"})
    inventory = {"scale_decoder_outputs": False, "shapes": {"shared.weight": [1, 1]}, "unique_parameters": 1,
                 "parameter_aliases": [["shared.weight", "encoder.embed_tokens.weight"]], "storage_aliases": []}
    parity = {"rtol": 1e-4, "atol": 1e-4, "max_abs_error": 0.}
    for role, score in (("student", 10.), ("teacher", 21.)):
        seal(role, "model", {
            "config.json": {"external_pair_profile": common.PROFILE, "scale_decoder_outputs": False,
                            "vocab_size": 32128, "decoder_start_token_id": 0,
                            "pad_token_id": 0, "eos_token_id": 1},
            "model.safetensors": b"tiny fp weights",
            "provenance.json": {"local_sft_updates": 0, "profile": common.PROFILE, "runtime": RUNTIME,
                                "origin": "external-public-sft",
                                "historical_steps": 100000 if role == "student" else 8000,
                                "historical_steps_status": "publisher-reported",
                                "source_complete_sha256": hashes[
                                    "public_student_source" if role == "student" else "teacher_source"],
                                "inventory": inventory, "reload_parity": parity,
                                "migration": parity, "fp32_bf16_drift": {"mean_forward_kl": 0.001}}},
            role=role, metadata={"role": role, "precision": "float32" if role == "student" else "bfloat16"})
        predictions = "".join(json.dumps({
            "id": id_, "reference": ref, "prediction": "summary",
            "scores": {key: score for key in ("rouge1", "rouge2", "rougeL")}}) + "\n"
            for id_, ref in zip(ids, refs))
        seal(role + "_validation", "validation", {
            "metrics.json": {"status": "complete", "n": VALIDATION_N, "split": "validation",
                             "precision": "bfloat16",
                              "scores": {key: score + metric_offsets.get(role, 0.)
                                         for key in ("rouge1", "rouge2", "rougeL")},
                             "counts": {"empty_prediction": 0, "hit_generation_cap": 0}, "runtime": RUNTIME,
                             "predictions_sha256": hashlib.sha256(predictions.encode()).hexdigest()},
            "predictions.jsonl": predictions}, role=role, model_complete_sha256=hashes[role],
            data_complete_sha256=hashes["prepared"], protocol=PROTOCOL)
    seal("quality", "quality", {"report.json": {
        "passed": 21. >= teacher_minimum, "n": VALIDATION_N,
        "scores": {"teacher": 21., "student": 10.}, "advantage": 11.,
        "paired_bootstrap_95ci": [10., 12.], "thresholds": definition["thresholds"]}},
        validation={role: hashes[role + "_validation"] for role in ("teacher", "student")})
    return root, definition
