"""Full held-out validation; no periodic training monitor."""
from __future__ import annotations

import json
import math
import time

from .common import (atomic_json, cached, digest, identity, pair_root, publish, read_json,
                     runtime, sha256, temporary, verify)
from .data import load as load_data, tokenizer as load_tokenizer
from .models import load_model
from sftlib.console import progress
from gridlib.policy import PROTOCOL, VALIDATION_N


def model_path(work, role):
    return pair_root(work) / "models" / f"{role}_{'fp32' if role == 'student' else 'bf16'}"


def evaluate(work, role, batch_size):
    import torch
    from rouge_score import rouge_scorer
    from sftlib.data import batch
    from sftlib.modeling import generate
    final = pair_root(work) / "validation" / role
    model_dir = model_path(work, role)
    verify(model_dir, identity(work, "model", role=role))
    expected = identity(work, "validation", role=role, model_complete_sha256=sha256(model_dir / "COMPLETE.json"),
                        data_complete_sha256=sha256(pair_root(work) / "student/prepared/COMPLETE.json"),
                        protocol=PROTOCOL)
    if cached(final, expected):
        return
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive validation batch required")
    device, info = runtime(work, gpu=True)
    model = load_model(model_dir, device, precision="bfloat16")
    tok = load_tokenizer(model_dir)
    dataset = load_data(work)["validation"]
    if len(dataset) != VALIDATION_N or len(set(map(str, dataset["id"]))) != VALIDATION_N:
        raise ValueError("Validation coverage/IDs changed")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {key: 0. for key in ("rouge1", "rouge2", "rougeL")}
    counts = {key: 0 for key in ("empty_prediction", "hit_generation_cap",
                                 "source_truncated", "reference_over_cap")}
    scratch = temporary(final)
    started = time.monotonic()
    with (scratch / "predictions.jsonl").open("x", encoding="utf-8") as stream, progress(
            VALIDATION_N, desc=f"External {role} validation", unit="summary") as bar:
        for start in range(0, VALIDATION_N, batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, VALIDATION_N))]
            inputs, labels = batch(tok, rows, device)
            ids = generate(model, tok, inputs, 64).cpu().tolist()
            if len(ids) != len(rows):
                raise ValueError("Generation batch coverage mismatch")
            for row, tokens in zip(rows, ids):
                eos = tok.eos_token_id in tokens
                if eos:
                    tokens = tokens[:tokens.index(tok.eos_token_id) + 1]
                text = tok.decode(tokens, skip_special_tokens=True).strip()
                scores = {key: score.fmeasure * 100 for key, score in scorer.score(row["reference"], text).items()}
                if any(not math.isfinite(value) for value in scores.values()):
                    raise FloatingPointError("Nonfinite validation scores")
                flags = {"empty_prediction": not text, "hit_generation_cap": len(tokens) >= 64 and not eos,
                         "source_truncated": bool(row["source_truncated"]),
                         "reference_over_cap": bool(row["reference_over_cap"])}
                for key in sums:
                    sums[key] += scores[key]
                for key in counts:
                    counts[key] += int(flags[key])
                stream.write(json.dumps({"id": str(row["id"]), "reference": row["reference"], "prediction": text,
                                          "generated_ids": tokens, "terminated_with_eos": eos,
                                          "scores": scores, **flags}, ensure_ascii=False) + "\n")
            stream.flush()
            del inputs, labels
            bar.update(len(rows))
    torch.cuda.synchronize(device)
    atomic_json(scratch / "metrics.json", {
        "status": "complete", "split": "validation", "n": VALIDATION_N,
        "scores": {key: value / VALIDATION_N for key, value in sums.items()},
        "counts": counts, "seconds": time.monotonic() - started, "batch_size": batch_size,
        "rates": {key: count / VALIDATION_N for key, count in counts.items()},
        "precision": "bfloat16", "runtime": info,
        "generation": {"max_new_tokens": 64, "num_beams": 1, "do_sample": False,
                       "min_new_tokens": 0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0},
        "predictions_sha256": sha256(scratch / "predictions.jsonl")})
    publish(scratch, final, expected)


def paired_rows(pair):
    metadata = read_json(pair / "student/prepared/data.json")
    outputs = {}
    for role in ("student", "teacher"):
        directory = pair / "validation" / role
        verify(directory)
        metrics = read_json(directory / "metrics.json")
        if (metrics.get("n") != VALIDATION_N or metrics.get("split") != "validation"
                or metrics.get("status") != "complete"
                or metrics.get("predictions_sha256") != sha256(directory / "predictions.jsonl")):
            raise ValueError("Incomplete/changed validation")
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = [row["id"] for row in rows]
        references = [row["reference"] for row in rows]
        if (len(rows) != VALIDATION_N or len(set(ids)) != VALIDATION_N
                or digest(ids) != metadata["ids_sha256"]["validation"]
                or digest(references) != metadata["validation_references_sha256"]):
            raise ValueError("Validation IDs/references differ from official prepared data")
        for key in ("rouge1", "rouge2", "rougeL"):
            values = [row["scores"][key] for row in rows]
            if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
                raise ValueError("Invalid per-example scores")
            aggregate = metrics["scores"][key]
            if type(aggregate) not in (int, float) or not math.isfinite(aggregate):
                raise ValueError("Nonfinite validation metrics")
        outputs[role] = rows
    if any(a["id"] != b["id"] or a["reference"] != b["reference"]
           for a, b in zip(outputs["student"], outputs["teacher"])):
        raise ValueError("Teacher/student evaluation pairing differs")
    return outputs


def bootstrap(differences):
    import numpy as np
    diffs = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(17)
    samples = []
    for _ in range(2000 // 20):
        indices = rng.integers(0, len(diffs), size=(20, len(diffs)))
        samples.extend(diffs[indices].mean(axis=1).tolist())
    return list(map(float, np.percentile(samples, [2.5, 97.5])))


def quality(work):
    final = pair_root(work) / "reports/quality"
    expected = identity(work, "quality", validation={
        role: sha256(pair_root(work) / "validation" / role / "COMPLETE.json") for role in ("teacher", "student")})
    if cached(final, expected):
        return
    rows = paired_rows(pair_root(work))
    low, high = bootstrap([t["scores"]["rouge2"] - s["scores"]["rouge2"]
                          for s, t in zip(rows["student"], rows["teacher"])])
    # paired_rows verified these aggregates against the saved per-example scores.
    # Use the persisted measurement as canonical: Python 3.12+ sum can round
    # differently from evaluate()'s incremental accumulation. Existing reports
    # remain immutable and are accepted using the same strict absolute tolerance.
    scores = {role: read_json(pair_root(work) / "validation" / role / "metrics.json")["scores"]["rouge2"]
              for role in rows}
    advantage = scores["teacher"] - scores["student"]
    scratch = temporary(final)
    atomic_json(scratch / "report.json", {
        "status": "complete", "policy": "informational-only", "scores": scores, "advantage": advantage,
        "paired_bootstrap_95ci": [low, high], "bootstrap_samples": 2000, "seed": 17,
        "n": VALIDATION_N,
        "publisher_teacher_r2": 21.370101928710938,
        "teacher_minus_publisher_r2": scores["teacher"] - 21.370101928710938,
        "historical_our_student_r2": 15.450918275676054,
        "note": "Informational measurements, not admission gates. Publisher scores are not our measured scores.",
        "empty_and_capped": {role: read_json(pair_root(work) / "validation" / role / "metrics.json")["counts"]
                             for role in scores}})
    publish(scratch, final, expected)
