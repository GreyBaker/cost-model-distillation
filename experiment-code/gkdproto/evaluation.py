from __future__ import annotations

import json
from pathlib import Path
import random
import time

from .common import (atomic_json, environment, load_rows, read_json, require_assets,
                     sha256, weight_signature, write_rows)
from .modeling import (autocast, device_and_precision, encode_sources, generate_targets,
                       load_model, load_tokenizer, target_mask)


def _truncated(tokenizer, row: dict, prefix: str, source_limit: int, target_limit: int) -> tuple[bool, bool]:
    return (len(tokenizer.encode(prefix + row["document"])) > source_limit,
            len(tokenizer.encode(row["reference"])) > target_limit)


def evaluate(args) -> None:
    import torch
    from rouge_score import rouge_scorer

    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation requires --allow-test; use validation while tuning")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    prepared = args.prepared_dir.resolve()
    manifest = read_json(prepared / "manifest.json")
    data_path = prepared / f"{args.split}.jsonl"
    if sha256(data_path) != manifest["subsets"][args.split]["sha256"]:
        raise ValueError("Prepared data hash changed")
    rows = load_rows(data_path)
    if args.limit:
        rows = rows[:args.limit]
    if args.batch_size < 1 or not rows:
        raise ValueError("Invalid evaluation batch/data size")
    device = device_and_precision(args.device, args.precision)
    tokenizer = load_tokenizer(args.model.resolve())
    model = load_model(args.model.resolve(), device, args.precision, trainable=False)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    output.mkdir(parents=True)
    atomic_json(output / "settings.json", {
        "model": str(args.model.resolve()), "model_signature": weight_signature(args.model.resolve()),
        "prepared_manifest": manifest, "split": args.split, "limit": args.limit,
        "device": str(device), "precision": args.precision, "prefix": args.prefix,
        "max_source_tokens": args.max_source_tokens, "max_target_tokens": args.max_target_tokens,
        "decoding": "greedy, beam=1", "rouge": "per-document F1 x100, macro mean, stemming",
        "environment": environment(),
        "model_provenance": getattr(args, "model_provenance", None),
        "evaluation_profile": getattr(args, "evaluation_profile", None),
        "evaluation_config_sha256": getattr(args, "evaluation_config_sha256", None),
        "seed": getattr(args, "seed", None),
    })
    sums = {key: 0.0 for key in ("rouge1", "rouge2", "rougeL")}
    counts = {"source_truncated": 0, "reference_over_target_cap": 0, "hit_generation_cap": 0, "empty_prediction": 0}
    started = time.perf_counter()
    # Stream predictions to disk so a long evaluation does not accumulate logits or rows.
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            sources = encode_sources(tokenizer, batch, args.prefix, args.max_source_tokens, device)
            with autocast(device, args.precision):
                ids = generate_targets(model, tokenizer, sources, args.max_target_tokens, sample=False)
            masks = target_mask(ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
            for row, tokens, mask in zip(batch, ids, masks):
                valid = tokens[mask].tolist()
                prediction = tokenizer.decode(valid, skip_special_tokens=True).strip()
                scores = {key: value.fmeasure * 100 for key, value in scorer.score(row["reference"], prediction).items()}
                for key, value in scores.items():
                    sums[key] += value
                source_truncated, reference_over_cap = _truncated(tokenizer, row, args.prefix, args.max_source_tokens, args.max_target_tokens)
                eos = tokenizer.eos_token_id in valid
                flags = {"source_truncated": source_truncated, "reference_over_target_cap": reference_over_cap,
                         "hit_generation_cap": len(valid) >= args.max_target_tokens and not eos,
                         "empty_prediction": not prediction}
                for key, value in flags.items():
                    counts[key] += int(value)
                stream.write(json.dumps({"id": row["id"], "split": args.split, "prediction": prediction,
                                         "reference": row["reference"], "target_tokens_including_eos": len(valid),
                                         "terminated_with_eos": eos, "scores": scores, **flags}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"Evaluated {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
            del sources, ids, masks
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    atomic_json(output / "metrics.json", {
        "status": "complete", "n": len(rows), "split": args.split,
        "scores": {key: value / len(rows) for key, value in sums.items()},
        "counts": counts, "seconds": time.perf_counter() - started,
        "predictions_sha256": sha256(output / "predictions.jsonl"),
        "warning": "ROUGE measures overlap, not faithfulness; this may be a small smoke subset.",
    })
    print(json.dumps(read_json(output / "metrics.json"), indent=2))


def cache_teacher(args) -> None:
    import torch
    from .common import seed_everything

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite teacher cache: {output}")
    require_assets(args.assets_dir.resolve())
    prepared = args.prepared_dir.resolve()
    prepared_manifest = read_json(prepared / "manifest.json")
    path = prepared / "train.jsonl"
    if sha256(path) != prepared_manifest["subsets"]["train"]["sha256"]:
        raise ValueError("Prepared train hash changed")
    rows = load_rows(path)
    device = device_and_precision(args.device, args.precision)
    seed_everything(args.seed)
    tokenizer = load_tokenizer(args.teacher.resolve())
    teacher = load_model(args.teacher.resolve(), device, args.precision, trainable=False)
    output.mkdir(parents=True)
    settings = {
        "status": "incomplete", "teacher": str(args.teacher.resolve()),
        "teacher_signature": weight_signature(args.teacher.resolve()),
        "train_sha256": sha256(path), "prefix": args.prefix,
        "max_source_tokens": args.max_source_tokens, "max_target_tokens": args.max_target_tokens,
        "precision": args.precision, "seed": args.seed, "decoding": "greedy, beam=1",
        "environment": environment(),
    }
    atomic_json(output / "manifest.json", settings)
    with (output / "targets.jsonl").open("x", encoding="utf-8") as stream:
        for number, row in enumerate(rows, 1):
            sources = encode_sources(tokenizer, [row], args.prefix, args.max_source_tokens, device)
            with autocast(device, args.precision):
                ids = generate_targets(teacher, tokenizer, sources, args.max_target_tokens, sample=False)
            mask = target_mask(ids, tokenizer.eos_token_id, tokenizer.pad_token_id)[0]
            tokens = ids[0][mask].tolist()
            stream.write(json.dumps({"id": row["id"], "target_ids": tokens,
                                     "text": tokenizer.decode(tokens, skip_special_tokens=True)}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"Cached teacher target {number}/{len(rows)}", flush=True)
            del sources, ids, mask
    settings.update(status="complete", count=len(rows), targets_sha256=sha256(output / "targets.jsonl"))
    atomic_json(output / "manifest.json", settings)


def compare(args) -> None:
    """Paired example bootstrap; not a substitute for multiple training seeds."""
    import numpy as np

    a = load_rows(args.a.resolve())
    b = load_rows(args.b.resolve())
    b_by_id = {row["id"]: row for row in b}
    if {row["id"] for row in a} != set(b_by_id):
        raise ValueError("Paired comparison requires identical evaluation IDs")
    if args.bootstrap_samples < 1:
        raise ValueError("Need positive bootstrap_samples")
    differences = []
    for row in a:
        other = b_by_id[row["id"]]
        if row["reference"] != other["reference"] or row["split"] != other["split"]:
            raise ValueError("References or splits differ")
        differences.append(row["scores"][args.metric] - other["scores"][args.metric])
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    samples = np.empty(args.bootstrap_samples)
    for i in range(args.bootstrap_samples):
        samples[i] = values[rng.integers(0, len(values), size=len(values))].mean()
    result = {"metric": args.metric, "direction": "A minus B, ROUGE points",
              "n": len(values), "mean_difference": float(values.mean()),
              "paired_bootstrap_95_percent_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
              "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
              "a_sha256": sha256(args.a.resolve()), "b_sha256": sha256(args.b.resolve()),
              "warning": "Example uncertainty only; independent-example assumption, no training-seed uncertainty."}
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
