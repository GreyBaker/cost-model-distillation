"""Plan a pinned XSum evaluation; --run explicitly enables model loading/scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from download_summarizer import MANIFEST, REPO, REVISION, verify_download
from gkdproto.common import load_rows, read_json, resolve, sha256

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "jordiclive_evaluation.json"
XSUM_REVISION = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"


def settings(profile: str, output_dir: Path | None = None) -> SimpleNamespace:
    config = read_json(CONFIG)
    if config["repo_id"] != REPO or config["revision"] != REVISION:
        raise ValueError("Evaluation/download checkpoint pins disagree")
    if config["split"] != "validation" or config["decoding"] != "greedy, beam=1":
        raise ValueError("This preparation is for greedy validation evaluation only")
    chosen = config["profiles"][profile]
    return SimpleNamespace(
        model=resolve(config["model_path"]), prepared_dir=resolve(chosen["prepared_dir"]),
        output_dir=output_dir.resolve() if output_dir else resolve(chosen["output_dir"]),
        split="validation", allow_test=False, limit=chosen["limit"],
        batch_size=config["batch_size"], device=config["device"], precision=config["precision"],
        prefix=config["prefix"], max_source_tokens=config["max_source_tokens"],
        max_target_tokens=config["max_target_tokens"], seed=config["seed"],
        expected_prepared_count=chosen["expected_prepared_count"],
        evaluation_profile=profile, evaluation_config_sha256=sha256(CONFIG),
    )


def check_data(args) -> None:
    manifest = read_json(args.prepared_dir / "manifest.json")
    if (manifest.get("status") != "complete"
            or manifest["dataset"].get("repo_id") != "EdinburghNLP/xsum"
            or manifest["dataset"].get("revision") != XSUM_REVISION):
        raise ValueError("Expected complete prepared data from the pinned official XSum revision")
    path = args.prepared_dir / "validation.jsonl"
    subset = manifest["subsets"]["validation"]
    if subset["count"] != args.expected_prepared_count or sha256(path) != subset["sha256"]:
        raise ValueError("Unexpected validation size or hash")
    rows = load_rows(path)
    if len(rows) != args.expected_prepared_count or any(row["split"] != "validation" for row in rows):
        raise ValueError("Validation rows/count disagree with the evaluation profile")


def run(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing evaluation output: {args.output_dir}")
    check_data(args)
    # Hash every downloaded file before unpickling the official .bin checkpoint.
    manifest = verify_download(args.model)
    args.model_provenance = {
        "repo_id": manifest["id"], "revision": manifest["sha"],
        "download_manifest_sha256": sha256(args.model / MANIFEST),
        "files": manifest["files"],
    }
    # Imports above are standard-library only; only explicit --run reaches the ML stack.
    from gkdproto.common import seed_everything
    from gkdproto.evaluation import evaluate
    seed_everything(args.seed)
    evaluate(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "validation"], default="smoke")
    parser.add_argument("--output-dir", type=Path, help="New output directory; never overwritten")
    parser.add_argument("--run", action="store_true", help="Verify assets and load/evaluate the model")
    args = parser.parse_args()
    resolved = settings(args.profile, args.output_dir)
    print(json.dumps(vars(resolved), indent=2, default=str))
    if args.run:
        run(resolved)
    else:
        print("Plan only: no weights, data tables or ML stack loaded. Add --run to evaluate.")


if __name__ == "__main__":
    main()
