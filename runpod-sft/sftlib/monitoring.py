"""Teacher greedy-quality diagnostics; best restart is separate from final 8K."""
from __future__ import annotations

import math
import random
import time

from . import checkpoints, data, modeling
from .core import atomic_json, code_signature, read_json, sha256
from .console import progress


def due(value, step):
    return bool(value.get("rouge_every_steps")) and (step == 0 or step % value["rouge_every_steps"] == 0)


def protocol(value):
    return {"split": "validation", "n": value["rouge_examples"],
            "selection_seed": value["seed"] + 1, "decoding": "greedy, beam=1",
            "max_source_tokens": value["max_source_tokens"], "max_target_tokens": value["max_target_tokens"],
            "precision": "FP32 training weights, BF16 autocast; not the rounded BF16 final export",
            "rouge": "full-reference stemmed per-document macro F1 x100"}


def finished(run_dir, value, step):
    if not due(value, step):
        return True
    path = run_dir / "monitoring" / f"step-{step:08d}.json"
    marker_path = path.with_suffix(".COMPLETE.json")
    if not marker_path.exists():
        return False
    marker = read_json(marker_path)
    run = read_json(run_dir / "run.json")
    result = read_json(path)
    if (marker.get("run_id") != run["run_id"] or marker.get("step") != step
            or marker.get("result_sha256") != sha256(path)
            or result.get("run_id") != run["run_id"] or result.get("step") != step
            or result.get("prepared_sha256") != run["signatures"]["prepared"]
            or result.get("source_checkpoint_complete_sha256") != sha256(
                run_dir / "checkpoints" / f"step-{step:08d}" / "COMPLETE.json")
            or result.get("protocol") != protocol(value)):
        raise ValueError("Changed/foreign completed teacher monitoring")
    if step > 0 and checkpoints.best_checkpoint(run_dir) is None:
        raise ValueError("Completed teacher monitoring lost its pinned trained-best restart")
    return True


def measure(model, tokenizer, dataset, value, device):
    import torch
    from rouge_score import rouge_scorer
    order = list(range(len(dataset)))
    random.Random(value["seed"] + 1).shuffle(order)
    order = order[:value["rouge_examples"]]
    if len(order) != value["rouge_examples"]:
        raise ValueError("Teacher monitoring requires the configured validation sample")
    batch_size = min(value["evaluation_batch_size"], value["microbatch_size"])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {name: 0.0 for name in ("rouge1", "rouge2", "rougeL")}
    counts = {"empty_prediction": 0, "hit_generation_cap": 0}
    examples = []
    started, was_training = time.monotonic(), model.training
    model.eval()
    try:
        with progress(len(order), desc=f"{value['role']} greedy monitor", unit="summary") as bar:
            for start in range(0, len(order), batch_size):
                rows = [dataset[index] for index in order[start:start + batch_size]]
                inputs, labels = data.batch(tokenizer, rows, device, limits=value)
                generated = modeling.generate(model, tokenizer, inputs, value["max_target_tokens"]).cpu().tolist()
                for row, ids in zip(rows, generated):
                    eos = tokenizer.eos_token_id in ids
                    if eos:
                        ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
                    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    scores = {name: score.fmeasure * 100
                              for name, score in scorer.score(row["reference"], text).items()}
                    flags = {"empty_prediction": not text,
                             "hit_generation_cap": len(ids) >= value["max_target_tokens"] and not eos}
                    for name in sums:
                        sums[name] += scores[name]
                    for name in counts:
                        counts[name] += int(flags[name])
                    examples.append({"id": row["id"], "reference": row["reference"],
                                     "prediction": text, "scores": scores, **flags})
                del inputs, labels
                bar.update(len(rows))
        torch.cuda.synchronize(device)
    finally:
        model.train(was_training)
    return {"scores": {name: total / len(order) for name, total in sums.items()},
            "counts": counts, "examples": examples, "n": len(order),
            "batch_size": batch_size, "seconds": time.monotonic() - started}


def record(run_dir, value, step, evaluate):
    """Publish diagnostics then pin best; replay repairs a crash between the two."""
    run = read_json(run_dir / "run.json")
    checkpoint = checkpoints.resolve(run_dir)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    if marker["step"] != step:
        raise ValueError("Teacher monitor must correspond to the latest complete restart")
    identity = {"run_id": run["run_id"], "step": step, "protocol": protocol(value),
                "source_checkpoint_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
                "prepared_sha256": run["signatures"]["prepared"]}
    path = run_dir / "monitoring" / f"step-{step:08d}.json"
    if path.exists():
        result = read_json(path)
        if any(result.get(key) != item for key, item in identity.items()):
            raise ValueError("Teacher monitor differs from its checkpoint/data/protocol")
    else:
        result = {**evaluate(), **identity, "code": code_signature()}
    if (result["n"] != value["rouge_examples"] or len(result["examples"]) != result["n"]
            or any(not math.isfinite(score) for score in result["scores"].values())):
        raise ValueError("Incomplete/nonfinite teacher greedy monitoring")
    if not path.exists():
        atomic_json(path, result)
    best_path = run_dir / "best_checkpoint.json"
    checkpoints.best_checkpoint(run_dir)  # Validate before comparing/updating an existing pin.
    best = read_json(best_path) if best_path.exists() else None
    # Exclude the unsupervised step-zero baseline from trained-best selection.
    if step > 0 and (best is None or result["scores"]["rouge2"] > best["rouge2"]):
        atomic_json(best_path, {"run_id": run["run_id"], "step": step, "path": checkpoint.name,
                    "rouge2": result["scores"]["rouge2"], "monitoring": path.relative_to(run_dir).as_posix(),
                    "monitoring_sha256": sha256(path),
                    "complete_sha256": identity["source_checkpoint_complete_sha256"],
                    "selection": "best fixed-subset monitoring R2; NOT an automatic replacement for final 8K"})
    if step == value["review_step"]:
        baseline_path = run_dir / "monitoring/step-00000000.json"
        baseline = read_json(baseline_path)["scores"]["rouge2"] if baseline_path.exists() else None
        atomic_json(run_dir / "teacher_review.json", {
            "step": step, "advisory_only": True, "monitoring": path.relative_to(run_dir).as_posix(),
            "baseline_rouge2": baseline, "current_rouge2": result["scores"]["rouge2"],
            "delta_from_base": result["scores"]["rouge2"] - baseline if baseline is not None else None,
            "note": "Inspect predictions, empty/capped outputs, CE and ROUGE before investing further. "
                    "No validated quality threshold or automatic early stop; final 8K remains required."})
    checkpoints.prune(run_dir)
    atomic_json(path.with_suffix(".COMPLETE.json"), {
        "run_id": run["run_id"], "step": step, "result_sha256": sha256(path)})
    return result
