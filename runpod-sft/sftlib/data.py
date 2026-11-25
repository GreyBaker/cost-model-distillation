from __future__ import annotations

import os
import uuid

from .core import atomic_json, check_space, data_signature, read_json, sha256, tree_manifest, verify_tree

COUNTS = {"train": 204045, "validation": 11332}


def prepare(work, value):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    target = work / "prepared"
    if target.exists():
        verify(work, value)
        print("Existing prepared train/validation verified; not overwritten.", flush=True)
        return
    check_space(work, 4 * 1024**3)
    temporary = work / (".prepared-" + uuid.uuid4().hex + ".incomplete")
    temporary.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(work / "assets/model", local_files_only=True, trust_remote_code=False)
    files = {split: str(work / "assets/xsum/data" / f"{split}-00000-of-00001.parquet") for split in COUNTS}
    dataset = load_dataset("parquet", data_files=files, cache_dir=str(work / "data_cache"))
    ids = {}
    for split, expected in COUNTS.items():
        ids[split] = set(map(str, dataset[split]["id"]))
        if len(dataset[split]) != expected or len(ids[split]) != expected:
            raise ValueError(f"Unexpected {split} count or duplicate IDs")
    if ids["train"] & ids["validation"]:
        raise ValueError("Official train/validation IDs overlap")

    def encode(batch, split):
        source = [value["prefix"] + text for text in batch["document"]]
        inputs = tokenizer(source, truncation=True, max_length=value["max_source_tokens"])
        labels = tokenizer(text_target=batch["summary"], truncation=True, max_length=value["max_target_tokens"])
        result = {"id": list(map(str, batch["id"])), "input_ids": inputs["input_ids"], "labels": labels["input_ids"]}
        if split == "validation":
            result.update(reference=batch["summary"],
                          source_truncated=[len(x) > value["max_source_tokens"] for x in tokenizer(source)["input_ids"]],
                          reference_over_cap=[len(x) > value["max_target_tokens"] for x in tokenizer(text_target=batch["summary"])["input_ids"]])
        return result

    for split in COUNTS:
        encoded = dataset[split].map(encode, batched=True, batch_size=128, writer_batch_size=128,
                                     fn_kwargs={"split": split}, remove_columns=dataset[split].column_names,
                                     load_from_cache_file=False)
        encoded.save_to_disk(str(temporary / split))
    atomic_json(temporary / "manifest.json", {"status": "complete",
                "assets_manifest_sha256": sha256(work / "assets/manifest.json"),
                "tokenization": data_signature(value), "counts": COUNTS, "files": tree_manifest(temporary)})
    os.replace(temporary, target)
    print(f"Prepared train/validation only: {target}", flush=True)


def verify(work, value):
    manifest = read_json(work / "prepared/manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("counts") != COUNTS
            or manifest.get("tokenization") != data_signature(value)
            or manifest.get("assets_manifest_sha256") != sha256(work / "assets/manifest.json")):
        raise ValueError("Prepared data incomplete or incompatible")
    verify_tree(work / "prepared", manifest["files"])
    return manifest


def load(work):
    from datasets import load_from_disk
    return {split: load_from_disk(str(work / "prepared" / split)) for split in COUNTS}


def target_count(rows):
    if not rows or any(not row["labels"] for row in rows):
        raise ValueError("Empty reference sequences")
    return sum(len(row["labels"]) for row in rows)


def batch(tokenizer, rows, device, limits=None):
    import torch
    # Inputs are intentionally pretokenized/cached. Re-encoding with __call__
    # would repeat work; silence only this fast-tokenizer padding advisory.
    notices = getattr(tokenizer, "deprecation_warnings", None)
    if isinstance(notices, dict):
        notices["Asking-to-pad-a-fast-tokenizer"] = True
    options = {"padding": True} if limits is None else {"padding": "max_length", "max_length": limits["max_source_tokens"]}
    inputs = tokenizer.pad([{"input_ids": row["input_ids"]} for row in rows],
                           return_attention_mask=True, return_tensors="pt", **options).to(device)
    width = limits["max_target_tokens"] if limits else max(len(row["labels"]) for row in rows)
    labels = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        if len(row["labels"]) > width:
            raise ValueError("Cached reference exceeds declared target cap")
        labels[index, :len(row["labels"])] = torch.tensor(row["labels"], dtype=torch.long, device=device)
    return dict(inputs), labels
