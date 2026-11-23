from __future__ import annotations

from pathlib import Path
import random

from .common import atomic_json, require_assets, sha256, write_rows

EXPECTED = {"train": 204045, "validation": 11332, "test": 11334}


def prepare(args) -> None:
    from datasets import load_dataset

    assets = args.assets_dir.resolve()
    manifest = require_assets(assets)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace a prepared dataset: {output}")
    # Parquet is read locally; no dataset remote Python scripts are executed.
    files = {split: str(assets / "xsum" / "data" / f"{split}-00000-of-00001.parquet") for split in EXPECTED}
    dataset = load_dataset("parquet", data_files=files, cache_dir=str(assets / "dataset_cache"))
    for split, expected in EXPECTED.items():
        if len(dataset[split]) != expected:
            raise ValueError(f"Unexpected XSum {split} count: {len(dataset[split])}")
    all_ids = {split: set(map(str, dataset[split]["id"])) for split in EXPECTED}
    for split, expected in EXPECTED.items():
        if len(all_ids[split]) != expected:
            raise ValueError(f"Duplicate XSum IDs in {split}")
    if any(all_ids[a] & all_ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("XSum split IDs overlap")
    output.mkdir(parents=True)
    details = {}
    for offset, split in enumerate(EXPECTED):
        requested = getattr(args, f"{split}_size")
        if requested < 1 or requested > len(dataset[split]):
            raise ValueError(f"Invalid {split} subset size: {requested}")
        order = list(range(len(dataset[split])))
        random.Random(args.subset_seed + offset).shuffle(order)
        selected = dataset[split].select(order[:requested])
        path = output / f"{split}.jsonl"
        write_rows(path, ({"id": str(row["id"]), "split": split,
                           "document": row["document"], "reference": row["summary"]} for row in selected))
        details[split] = {"count": requested, "sha256": sha256(path)}
    atomic_json(output / "manifest.json", {
        "status": "complete", "dataset": manifest["assets"]["xsum"],
        "subset_seed": args.subset_seed, "official_counts": EXPECTED, "subsets": details,
        "note": "Seeded subsets of official splits. No test inputs used for training.",
    })
    print(f"Prepared local XSum subsets in {output}")
