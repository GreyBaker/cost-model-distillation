"""Full, greedy validation with atomic completion marker and retry preservation."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .policy import PROTOCOL, VALIDATION_N
from .shared import (atomic_json, checkpoints, read_json, sha256, tree_manifest,
                     verify_tree)
from sftlib.console import output as display, progress


def verify(output, identity):
    marker = read_json(output / "COMPLETE.json")
    if marker.get("identity") != identity or marker.get("n") != VALIDATION_N:
        raise ValueError("Evaluation identity/coverage changed")
    verify_tree(output, marker["files"])
    metrics = read_json(output / "metrics.json")
    if (metrics.get("status") != "complete" or metrics.get("n") != VALIDATION_N
            or metrics.get("split") != "validation"
            or metrics["predictions_sha256"] != sha256(output / "predictions.jsonl")):
        raise ValueError("Evaluation incomplete or prediction hash changed")
    return metrics


def source(root, bound, item):
    if item is None:
        directory = Path(bound["inputs"]["paths"]["student_checkpoint"])
    elif "trajectory_id" in item:
        from .controller import saved_pilot
        from .milestones import verify
        directory, _ = verify(root, bound, item, saved_pilot(root))
    else:
        run_dir = root / "cells" / item["id"]
        result = read_json(run_dir / "training_summary.json")
        if result.get("status") != "complete" or result["cell"] != item:
            raise ValueError("Evaluate only a completed matching cell")
        directory = checkpoints.resolve(run_dir)
        if str(directory) != result["checkpoint"]:
            raise ValueError("Training result does not refer to latest cell checkpoint")
        checkpoints.verify(directory, read_json(run_dir / "run.json")["run_id"])
    return directory


def identity(root, bound, item, selected):
    from .recovery import evaluation_microbatch
    directory = source(root, bound, item)
    result = {"source_complete_sha256": sha256(directory / "COMPLETE.json"),
            "binding_sha256": sha256(root / "binding.json"),
            "protocol": PROTOCOL, "batch_size": evaluation_microbatch(root, bound, item, selected),
             "precision": "bfloat16", "split": "validation", "n": VALIDATION_N}
    if (root / "evaluation_pilot.json").exists():
        result["evaluation_pilot_sha256"] = sha256(root / "evaluation_pilot.json")
    return result


def evaluate(root, bound, item, selected, output):
    started = time.monotonic()
    expected = identity(root, bound, item, selected)
    if output.exists():
        return verify(output, expected)
    if not (root / "evaluation_pilot.json").exists():
        raise ValueError("Run the once-only evaluation batch pilot before new evaluations")
    import torch
    from rouge_score import rouge_scorer
    from sftlib import data, modeling
    from . import backend
    directory = source(root, bound, item)
    verify_tree(directory, read_json(directory / "COMPLETE.json")["files"])
    device, runtime = backend.runtime(bound["settings"])
    if runtime != selected["runtime"]:
        raise ValueError("Evaluation runtime differs from the selected pilot")
    tokenizer, model = modeling.load(directory, device, False, {**PROTOCOL, "gradient_checkpointing": False})
    dataset = data.load(Path(bound["inputs"]["sft_work"]) / "student")["validation"]
    ids = list(map(str, dataset["id"]))
    if len(dataset) != VALIDATION_N or len(set(ids)) != VALIDATION_N:
        raise ValueError("Wrong validation coverage or duplicate IDs")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    output.mkdir(parents=True)
    atomic_json(output / "settings.json", {"identity": expected, "runtime": runtime,
                "rouge": "full-reference stemmed per-document macro F1 x100"})
    sums = {name: 0.0 for name in ("rouge1", "rouge2", "rougeL")}
    counters = {name: 0 for name in ("source_truncated", "reference_over_cap",
                                    "hit_generation_cap", "empty_prediction")}
    torch.cuda.synchronize(device)
    generation_started = time.monotonic()
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream, \
            progress(VALIDATION_N, desc=f"{item['id'] if item else 'SFT baseline'} validation",
                     unit="summary") as bar:
        for start in range(0, VALIDATION_N, expected["batch_size"]):
            rows = [dataset[i] for i in range(start, min(VALIDATION_N, start + expected["batch_size"]))]
            inputs, _ = data.batch(tokenizer, rows, device)
            generated = backend.generate(model, tokenizer, inputs, sample=False).cpu().tolist()
            if len(generated) != len(rows):
                raise ValueError("Generation batch coverage mismatch")
            for row, tokens in zip(rows, generated):
                eos = tokenizer.eos_token_id in tokens
                if eos:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                scores = {name: value.fmeasure * 100 for name, value in scorer.score(row["reference"], text).items()}
                flags = {"source_truncated": bool(row["source_truncated"]),
                         "reference_over_cap": bool(row["reference_over_cap"]),
                         "hit_generation_cap": len(tokens) >= 64 and not eos,
                         "empty_prediction": not text}
                for name in sums:
                    sums[name] += scores[name]
                for name in counters:
                    counters[name] += int(flags[name])
                stream.write(json.dumps({"id": str(row["id"]), "split": "validation",
                             "reference": row["reference"], "prediction": text, "generated_ids": tokens,
                             "terminated_with_eos": eos, "scores": scores, **flags}, ensure_ascii=False) + "\n")
            stream.flush()
            bar.set_postfix(R2=f"{sums['rouge2'] / (start + len(rows)):.2f}", refresh=False)
            bar.update(len(rows))
    torch.cuda.synchronize(device)
    metrics = {"status": "complete", "split": "validation", "n": VALIDATION_N,
               "scores": {name: score / VALIDATION_N for name, score in sums.items()},
               "counts": counters, "evaluation_seconds": time.monotonic() - generation_started,
               "wall_seconds": time.monotonic() - started,
               "predictions_sha256": sha256(output / "predictions.jsonl")}
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "COMPLETE.json", {"identity": expected, "n": VALIDATION_N,
                                         "files": tree_manifest(output)})
    display({"scores": metrics["scores"], "evaluation_seconds": metrics["evaluation_seconds"]},
            f"{item['id'] if item else 'SFT baseline'} validation complete")
    return metrics
