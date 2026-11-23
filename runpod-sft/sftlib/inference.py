from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

from . import checkpoints, data, modeling
from .core import (STEPS, atomic_json, check_space, code_signature, environment, read_json, sha256,
                   sync_directory, sync_tree, tree_manifest, verify_tree)
from .resume import config_compatible


def export_path(work, value, step=None):
    chosen = value["max_steps"] if step is None else step
    return work / "exports" / f"{value['role']}_xsum_step{chosen:08d}_bf16"


def verify_export(directory):
    marker = read_json(directory / "COMPLETE.json")
    if (marker.get("kind") != "sft-bf16-inference" or marker.get("status") != "complete"
            or marker.get("role") not in STEPS):
        raise ValueError("Not a complete portable SFT BF16 export")
    verify_tree(directory, marker["files"])
    provenance = read_json(directory / "provenance.json")
    if provenance["source_run"]["config"]["role"] != marker["role"] or provenance["step"] != marker["step"]:
        raise ValueError("Export role/step provenance mismatch")
    return marker


def probe(model, tokenizer, rows, device, value):
    import torch
    inputs, _ = data.batch(tokenizer, rows, device)
    decoder = torch.full((len(rows), 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, decoder_input_ids=decoder, use_cache=False).logits[:, 0].float().cpu()
    tokens = modeling.generate(model, tokenizer, inputs, value["max_target_tokens"]).cpu()
    return logits, tokens, {name: tensor.cpu().tolist() for name, tensor in inputs.items()}


def export(work, value, allow_partial=False):
    import torch
    from safetensors.torch import save_file
    run_dir = work / "runs" / value["run_name"]
    run = read_json(run_dir / "run.json")
    if not config_compatible(run["config"], value):
        raise ValueError("Export configuration changed")
    checkpoint = checkpoints.resolve(run_dir)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    if marker["step"] != value["max_steps"] and not allow_partial:
        raise ValueError("Final export requires the full role-specific update budget")
    data.verify(work, value)
    if sha256(work / "prepared/manifest.json") != run["signatures"]["prepared"]:
        raise ValueError("Prepared data changed since training")
    final = export_path(work, value, marker["step"])
    if final.exists():
        existing = verify_export(final)
        provenance = read_json(final / "provenance.json")
        if (existing["step"] != marker["step"] or provenance["source_run"] != run
                or provenance["source_checkpoint_complete_sha256"] != sha256(checkpoint / "COMPLETE.json")):
            raise ValueError("Existing export belongs to a different source checkpoint")
        print(f"Existing export verified: {final}", flush=True)
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    model_bytes = sum(entry["bytes"] for name, entry in marker["files"].items() if name.endswith(".safetensors"))
    check_space(final.parent, int(model_bytes * 0.6) + 1024**3)
    temporary = final.parent / ("." + final.name + "." + uuid.uuid4().hex + ".incomplete")
    temporary.mkdir()
    device = modeling.cuda_device(value, for_training=False)
    tokenizer, model = modeling.load(checkpoint, device, False, value)
    validation = data.load(work)["validation"]
    rows = [validation[index] for index in range(2)]
    before, tokens_before, inputs = probe(model, tokenizer, rows, device, value)
    model.config.use_cache = True
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(temporary)
    del model
    torch.cuda.empty_cache()
    tokenizer, reloaded = modeling.load(temporary, device, False, value)
    after, tokens_after, _ = probe(reloaded, tokenizer, rows, device, value)
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)
    if not torch.equal(tokens_before, tokens_after):
        raise ValueError("BF16 export/reload generation parity failed")
    (temporary / "checks").mkdir()
    save_file({"first_token_logits": after.contiguous()}, temporary / "checks/probe.safetensors")
    atomic_json(temporary / "checks/probe.json", {"ids": [row["id"] for row in rows], "inputs": inputs,
                "generated_ids": tokens_after.tolist(), "logit_max_abs_difference": float((before - after).abs().max()),
                "comparison": "same-device BF16 serialization parity, not FP32/BF16 quality equivalence"})
    atomic_json(temporary / "provenance.json", {"source_run": run, "step": marker["step"],
                "source_checkpoint_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
                "export_environment": environment(), "export_code": code_signature(),
                "export_precision": "bfloat16", "partial": marker["step"] != value["max_steps"]})
    (temporary / "README.md").write_text(
        f"# XSum-SFT {value['role']}\n\n{marker['step']} added optimizer updates from a pinned LM-adapted base. "
        "Reconstructed SFT recipe, not the authors' exact checkpoint. BF16 safetensors for inference. "
        "Keep the separate FP32 checkpoint for full optimizer resume or highest-fidelity student initialization.\n",
        encoding="utf-8")
    sync_tree(temporary)
    atomic_json(temporary / "COMPLETE.json", {"kind": "sft-bf16-inference", "status": "complete",
                "role": value["role"], "step": marker["step"], "files": tree_manifest(temporary)})
    sync_directory(temporary)
    os.replace(temporary, final)
    sync_directory(final.parent)
    print(f"Verified BF16 export: {final}", flush=True)
    return final


def verify_evaluation(work, value, directory, export_dir, expected_n):
    settings = read_json(directory / "settings.json")
    metrics = read_json(directory / "metrics.json")
    if (not config_compatible(settings["config"], value) or settings["export_complete_sha256"] != sha256(export_dir / "COMPLETE.json")
            or settings["prepared_manifest_sha256"] != sha256(work / "prepared/manifest.json")
            or metrics["status"] != "complete" or metrics["split"] != "validation" or metrics["n"] != expected_n
            or sha256(directory / "predictions.jsonl") != metrics["predictions_sha256"]):
        raise ValueError("Evaluation differs from the final export/data/protocol")
    return metrics


def evaluate(work, value, output, limit=0):
    import torch
    from rouge_score import rouge_scorer
    if type(limit) is not int or not 0 <= limit <= data.COUNTS["validation"]:
        raise ValueError("Invalid validation limit; 0 means full validation")
    export_dir = export_path(work, value)
    marker = verify_export(export_dir)
    provenance = read_json(export_dir / "provenance.json")
    if (marker["role"] != value["role"] or marker["step"] != value["max_steps"]
            or not config_compatible(provenance["source_run"]["config"], value) or provenance["partial"]):
        raise ValueError("Evaluate the complete matching role/config export")
    data.verify(work, value)
    count = limit or data.COUNTS["validation"]
    if output.exists():
        metrics = verify_evaluation(work, value, output, export_dir, count)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
    dataset = data.load(work)["validation"]
    device = modeling.cuda_device(value, for_training=False)
    tokenizer, model = modeling.load(export_dir, device, False, value)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    output.mkdir(parents=True)
    atomic_json(output / "settings.json", {"config": value, "export_complete_sha256": sha256(export_dir / "COMPLETE.json"),
                "prepared_manifest_sha256": sha256(work / "prepared/manifest.json"), "split": "validation", "n": count,
                "environment": environment(), "decoding": "greedy, beam=1", "precision": "bfloat16",
                "batch_size": value["evaluation_batch_size"], "rouge": "full-reference stemmed per-document macro F1 x100"})
    sums = {name: 0.0 for name in ("rouge1", "rouge2", "rougeL")}
    counters = {name: 0 for name in ("source_truncated", "reference_over_cap", "hit_generation_cap", "empty_prediction")}
    started = time.monotonic()
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for start in range(0, count, value["evaluation_batch_size"]):
            rows = [dataset[index] for index in range(start, min(count, start + value["evaluation_batch_size"]))]
            inputs, _ = data.batch(tokenizer, rows, device)
            generated = modeling.generate(model, tokenizer, inputs, value["max_target_tokens"]).cpu().tolist()
            for row, ids in zip(rows, generated):
                eos = tokenizer.eos_token_id in ids
                if eos:
                    ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                scores = {name: score.fmeasure * 100 for name, score in scorer.score(row["reference"], text).items()}
                flags = {"source_truncated": bool(row["source_truncated"]), "reference_over_cap": bool(row["reference_over_cap"]),
                         "hit_generation_cap": len(ids) >= value["max_target_tokens"] and not eos, "empty_prediction": not text}
                for name in sums:
                    sums[name] += scores[name]
                for name in counters:
                    counters[name] += int(flags[name])
                stream.write(json.dumps({"id": row["id"], "split": "validation", "reference": row["reference"],
                             "prediction": text, "target_tokens_including_eos": len(ids), "terminated_with_eos": eos,
                             "scores": scores, **flags}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"Evaluated {min(start + len(rows), count)}/{count}", flush=True)
    torch.cuda.synchronize(device)
    metrics = {"status": "complete", "role": value["role"], "split": "validation", "n": count,
               "scores": {name: result / count for name, result in sums.items()}, "counts": counters,
               "seconds": time.monotonic() - started, "predictions_sha256": sha256(output / "predictions.jsonl")}
    atomic_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
