from __future__ import annotations

import json
import os
import time
import uuid

from . import checkpoints, data, modeling
from .core import (atomic_json, check_space, code_signature, environment, read_json, sha256,
                   sync_directory, sync_tree, tree_manifest, verify_tree)


def verify_export(directory):
    marker = read_json(directory / "COMPLETE.json")
    if marker.get("kind") != "teacher-inference" or marker.get("status") != "complete":
        raise ValueError("Not a complete portable teacher export")
    verify_tree(directory, marker["files"])
    return marker


def probe(model, tokenizer, rows, device, config):
    import torch
    inputs, _ = data.batch(tokenizer, rows, device)
    decoder = torch.full((len(rows), 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, decoder_input_ids=decoder, use_cache=False).logits[:, 0].float().cpu()
    tokens = modeling.generate(model, tokenizer, inputs, config["max_target_tokens"]).cpu()
    return logits, tokens, {key: value.cpu().tolist() for key, value in inputs.items()}


def export(work, config, selection="latest", allow_partial=False):
    import torch
    from safetensors.torch import save_file

    run_dir = work / "runs" / config["run_name"]
    run = read_json(run_dir / "run.json")
    if run["config"] != config:
        raise ValueError("Export config differs from training run")
    checkpoint = checkpoints.resolve_checkpoint(run_dir, selection)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    if marker["step"] != 8000 and not allow_partial:
        raise ValueError("Final export requires step 8000; --allow-partial is for labeled pilot artifacts")
    data.verify(work, config)
    if sha256(work / "prepared" / "manifest.json") != run["signatures"]["prepared"]:
        raise ValueError("Prepared data differs from the trained run")
    parent = work / "exports"
    parent.mkdir(exist_ok=True)
    final = parent / f"teacher_xsum_step{marker['step']:08d}_bf16"
    if final.exists():
        raise FileExistsError(f"Export already exists; verify/use it instead: {final}")
    check_space(parent, 7 * 1024 ** 3)
    temporary = parent / ("." + final.name + "." + uuid.uuid4().hex + ".incomplete")
    temporary.mkdir()
    # Export/evaluation need native BF16, but do not enforce the training VRAM floor.
    device = modeling.cuda_device(config, for_training=False)
    tokenizer, model = modeling.load(checkpoint, device, False, config)
    validation = data.load(work)["validation"]
    rows = [validation[index] for index in range(2)]
    before, tokens_before, probe_inputs = probe(model, tokenizer, rows, device, config)
    model.config.use_cache = True
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(temporary)
    del model
    torch.cuda.empty_cache()
    tokenizer, reloaded = modeling.load(temporary, device, False, config)
    after, tokens_after, _ = probe(reloaded, tokenizer, rows, device, config)
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)
    if not torch.equal(tokens_before, tokens_after):
        raise ValueError("BF16 export/reload generation parity check failed")
    checks = temporary / "checks"
    checks.mkdir()
    save_file({"first_token_logits": after.contiguous()}, checks / "probe.safetensors")
    atomic_json(checks / "probe.json", {"ids": [row["id"] for row in rows], "inputs": probe_inputs,
                "generated_ids": tokens_after.tolist(), "logit_max_abs_difference": float((before - after).abs().max()),
                "comparison": "same-device BF16 before/after serialization, not FP32-vs-BF16 equivalence"})
    atomic_json(temporary / "provenance.json", {"source_run": run, "step": marker["step"],
                "source_checkpoint_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
                "export_environment": environment(), "export_code": code_signature(),
                "export_precision": "bfloat16", "partial": marker["step"] != 8000})
    (temporary / "README.md").write_text(
        f"# XSum-SFT LM-adapted T5-XL\n\nFull-parameter SFT, {marker['step']} optimizer updates. "
        "Base/dataset revisions, configuration and restart provenance are in provenance.json. "
        "This is a reconstructed preparation recipe, not the authors' exact checkpoint. "
        "Weights exported as BF16 safetensors; not quantized or adapter-only. "
        "ROUGE must be measured separately.\n", encoding="utf-8")
    sync_tree(temporary)
    atomic_json(temporary / "COMPLETE.json", {"kind": "teacher-inference", "status": "complete",
                "step": marker["step"], "files": tree_manifest(temporary)})
    sync_directory(temporary)
    os.replace(temporary, final)
    sync_directory(parent)
    print(f"Portable teacher exported with reload parity checks: {final}")


def evaluate(work, config, export_dir, output, limit=256):
    import torch
    from rouge_score import rouge_scorer

    if output.exists():
        raise FileExistsError(f"Never overwrite evaluation output: {output}")
    verify_export(export_dir)
    provenance = read_json(export_dir / "provenance.json")
    if provenance["source_run"]["config"] != config:
        raise ValueError("Evaluation configuration differs from exported teacher protocol")
    data.verify(work, config)
    dataset = data.load(work)["validation"]
    if limit < 0 or limit > len(dataset):
        raise ValueError("Invalid validation limit (0 means full validation)")
    count = limit or len(dataset)
    device = modeling.cuda_device(config, for_training=False)
    tokenizer, model = modeling.load(export_dir, device, False, config)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    output.mkdir(parents=True)
    atomic_json(output / "settings.json", {"config": config, "export_provenance": provenance,
                "export_complete_sha256": sha256(export_dir / "COMPLETE.json"),
                "prepared_manifest_sha256": sha256(work / "prepared" / "manifest.json"),
                "split": "validation", "n": count, "decoding": "greedy, beam=1, no sampling/length penalties",
                "environment": environment(),
                "rouge": "full reference; stemmed per-document macro F1 x100", "precision": "bfloat16"})
    sums = {key: 0.0 for key in ("rouge1", "rouge2", "rougeL")}
    counters = {key: 0 for key in ("source_truncated", "reference_over_cap", "hit_generation_cap", "empty_prediction")}
    started = time.monotonic()
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(count):
            row = dataset[index]
            inputs, _ = data.batch(tokenizer, [row], device)
            tokens = modeling.generate(model, tokenizer, inputs, config["max_target_tokens"])[0].tolist()
            eos = tokenizer.eos_token_id in tokens
            if eos:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            scores = {key: score.fmeasure * 100 for key, score in scorer.score(row["reference"], text).items()}
            flags = {"source_truncated": bool(row["source_truncated"]), "reference_over_cap": bool(row["reference_over_cap"]),
                     "hit_generation_cap": len(tokens) >= config["max_target_tokens"] and not eos, "empty_prediction": not text}
            for key in sums:
                sums[key] += scores[key]
            for key in counters:
                counters[key] += int(flags[key])
            stream.write(json.dumps({"id": row["id"], "split": "validation", "reference": row["reference"],
                         "prediction": text, "scores": scores, "terminated_with_eos": eos, **flags}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"Evaluated {index + 1}/{count}", flush=True)
    atomic_json(output / "metrics.json", {"status": "complete", "split": "validation", "n": count,
                "scores": {key: value / count for key, value in sums.items()}, "counts": counters,
                "seconds": time.monotonic() - started, "predictions_sha256": sha256(output / "predictions.jsonl")})
    print(json.dumps(read_json(output / "metrics.json"), indent=2))
