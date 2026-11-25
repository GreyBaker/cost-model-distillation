"""Official local parquet data and cross-runtime tokenizer audit."""
from __future__ import annotations

import hashlib
import json
import os
import random

from .acquire import source, student_source
from .common import (atomic_json, cached, digest, identity, pair_root, publish, read_json,
                     request, runtime, sha256, temporary, verify)
from sftlib.console import progress
from sftlib.data import COUNTS
from gridlib.policy import PROTOCOL

EDGES = ["  repeated\tspaces\nand newlines  ", "café cafe\u0301 naïve Straße",
         "“Quotes”—dashes… punctuation!", "中文 العربية 🙂", "<extra_id_0> <extra_id_99>",
         "A " * 1100, "", "line\r\nending\u00a0space"]


def raw(work):
    from datasets import load_dataset
    directory = source(work, "xsum")
    verify(directory)
    dataset = load_dataset("parquet", data_files={
        split: str(directory / "data" / f"{split}-00000-of-00001.parquet") for split in COUNTS})
    ids = {}
    for split, count in COUNTS.items():
        ids[split] = set(map(str, dataset[split]["id"]))
        if len(dataset[split]) != count or len(ids[split]) != count:
            raise ValueError("Unexpected official data count or duplicate IDs")
    if ids["train"] & ids["validation"]:
        raise ValueError("Official split IDs overlap")
    return dataset


def corpus(work):
    final = pair_root(work) / "reports/corpus"
    expected = identity(work, "corpus")
    if cached(final, expected):
        return
    dataset = raw(work)
    indices = sorted(random.Random(17).sample(range(COUNTS["train"]), 2048))
    scratch = temporary(final)
    with (scratch / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for i, text in enumerate(EDGES):
            stream.write(json.dumps({"id": f"edge-{i}", "split": "edge", "document": text, "summary": text},
                                    ensure_ascii=False) + "\n")
        for split, chosen in (("validation", range(COUNTS["validation"])), ("train", indices)):
            for index in chosen:
                row = dataset[split][index]
                stream.write(json.dumps({"id": str(row["id"]), "split": split,
                                         "document": row["document"], "summary": row["summary"]},
                                        ensure_ascii=False) + "\n")
    publish(scratch, final, expected)


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def audit_tokenizer(work, path):
    tok = tokenizer(path)
    if (tok.pad_token_id, tok.eos_token_id, tok.unk_token_id) != (0, 1, 2):
        raise ValueError("Unexpected tokenizer PAD/EOS/UNK semantics")
    vocab = tok.get_vocab()
    ids = list(vocab.values())
    if (not ids or any(type(value) is not int or not 0 <= value < 32128 for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Tokenizer IDs must be unique and fit all model output rows without resizing")
    original_max = tok.model_max_length
    tok.model_max_length = 10 ** 9  # Audit full encodings; avoid the misleading saved 512-token advisory.
    encoding, decoding, counts = hashlib.sha256(), hashlib.sha256(), {}
    corpus_dir = pair_root(work) / "reports/corpus"
    verify(corpus_dir, identity(work, "corpus"))
    with (corpus_dir / "rows.jsonl").open(encoding="utf-8") as stream, progress(
            len(EDGES) + 11332 + 2048, desc="Tokenizer audit", unit="row") as bar:
        for line in stream:
            row = json.loads(line)
            src = tok(PROTOCOL["prefix"] + row["document"], truncation=False)["input_ids"]
            target = tok(text_target=row["summary"], truncation=False)["input_ids"]
            capped_src = tok(PROTOCOL["prefix"] + row["document"], truncation=True, max_length=1024)["input_ids"]
            capped_target = tok(text_target=row["summary"], truncation=True, max_length=64)["input_ids"]
            encoding.update(json.dumps([row["split"], row["id"], src, target, capped_src, capped_target],
                                       separators=(",", ":")).encode())
            decoding.update(json.dumps([tok.decode(src, skip_special_tokens=True),
                                        tok.decode(target, skip_special_tokens=True)],
                                       ensure_ascii=False, separators=(",", ":")).encode())
            counts[row["split"]] = counts.get(row["split"], 0) + 1
            bar.update()
    return {"vocab_sha256": digest(vocab), "vocabulary_entries": len(vocab),
            "max_token_id": max(ids), "padded_output_rows": 32128 - max(ids) - 1,
            "encoding_sha256": encoding.hexdigest(),
            "decoding_sha256": decoding.hexdigest(), "special_ids": {
                name: getattr(tok, name) for name in ("pad_token_id", "eos_token_id", "unk_token_id")},
            "special_tokens_map": {k: str(v) for k, v in tok.special_tokens_map.items()},
            "original_model_max_length": original_max, "counts": counts}


def token_audit(work, reference=False):
    runtime(work, reference=reference)
    final = pair_root(work) / "reports" / ("legacy_tokenizer" if reference else "tokenizers")
    expected = identity(work, "legacy_tokenizer" if reference else "tokenizers")
    if cached(final, expected):
        return
    student = audit_tokenizer(work, student_source(work))
    results = {"student": student}
    if not reference:
        results["teacher"] = audit_tokenizer(work, source(work, "teacher"))
        peers = [results["teacher"]]
        if request(work)["definition"]["student"] == "restored":
            legacy = pair_root(work) / "reports/legacy_tokenizer"
            verify(legacy, identity(work, "legacy_tokenizer"))
            peers.append(read_json(legacy / "report.json")["student"])
        keys = ("vocab_sha256", "encoding_sha256", "decoding_sha256", "special_ids", "counts")
        for peer in peers:
            if any(student[key] != peer[key] for key in keys):
                raise ValueError("Tokenizer mapping/normalization/decode mismatch. Stop; explicitly select public "
                                 "student in a NEW work/grid root if needed. No automatic swap or cache edit.")
    scratch = temporary(final)
    atomic_json(scratch / "report.json", results)
    publish(scratch, final, expected)


def prepare(work):
    runtime(work)
    verify(pair_root(work) / "reports/tokenizers", identity(work, "tokenizers"))
    final = pair_root(work) / "student/prepared"
    expected = identity(work, "prepared")
    if cached(final, expected):
        return
    dataset = raw(work)
    tok = tokenizer(student_source(work))
    tok.model_max_length = 10 ** 9
    scratch = temporary(final)

    def encode(batch, split):
        text = [PROTOCOL["prefix"] + x for x in batch["document"]]
        inputs = tok(text, truncation=True, max_length=1024)["input_ids"]
        targets = tok(text_target=batch["summary"], truncation=True, max_length=64)["input_ids"]
        result = {"id": list(map(str, batch["id"])), "input_ids": inputs, "labels": targets}
        if split == "validation":
            result.update(reference=batch["summary"],
                          source_truncated=[len(x) > 1024 for x in tok(text)["input_ids"]],
                          reference_over_cap=[len(x) > 64 for x in tok(text_target=batch["summary"])["input_ids"]])
        return result

    ids_hashes = {}
    for split, count in COUNTS.items():
        encoded = dataset[split].map(encode, batched=True, batch_size=128, writer_batch_size=128,
                                      fn_kwargs={"split": split}, remove_columns=dataset[split].column_names,
                                      load_from_cache_file=False)
        if len(encoded) != count:
            raise ValueError("Tokenization changed dataset coverage")
        encoded.save_to_disk(str(scratch / split))
        ids_hashes[split] = digest(list(map(str, encoded["id"])))
    atomic_json(scratch / "data.json", {"counts": COUNTS, "protocol": PROTOCOL,
                "ids_sha256": ids_hashes, "source_complete_sha256": sha256(source(work, "xsum") / "COMPLETE.json"),
                "validation_references_sha256": digest(list(dataset["validation"]["summary"])),
                "tokenizer_report_sha256": sha256(pair_root(work) / "reports/tokenizers/COMPLETE.json")})
    publish(scratch, final, expected)


def load(work):
    from sftlib.data import load as load_prepared
    directory = pair_root(work) / "student/prepared"
    verify(directory, identity(work, "prepared"))
    return load_prepared(pair_root(work) / "student")
