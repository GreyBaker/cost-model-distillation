"""CLI entry point. --help itself does not import the ML stack or load models."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def generation_arguments(parser, model_option: str):
    parser.add_argument(model_option, type=Path, default=ROOT / "assets" / ("teacher" if model_option == "--teacher" else "student"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--prefix", default="summarize: ")
    parser.add_argument("--max-source-tokens", type=positive, default=256)
    parser.add_argument("--max-target-tokens", type=positive, default=32)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "prepared" / "smoke")
    parser.add_argument("--output-dir", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description="T5-small/XL XSum GKD prototype; no subcommand runs automatically.")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Make deterministic subsets from downloaded local XSum parquet")
    prep.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    prep.add_argument("--output-dir", type=Path, default=ROOT / "prepared" / "smoke")
    prep.add_argument("--train-size", type=positive, default=256)
    prep.add_argument("--validation-size", type=positive, default=64)
    prep.add_argument("--test-size", type=positive, default=64)
    prep.add_argument("--subset-seed", type=int, default=20261106)
    cache = commands.add_parser("cache-teacher", help="Freeze greedy teacher targets on prepared train IDs")
    generation_arguments(cache, "--teacher")
    cache.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    cache.add_argument("--seed", type=int, default=17)
    train = commands.add_parser("train", help="Run one independent GKD/SFT cell; supports its own checkpoint resume")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--resume", type=Path)
    evaluation = commands.add_parser("evaluate", help="Generate student summaries and score per-document ROUGE")
    generation_arguments(evaluation, "--model")
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--allow-test", action="store_true")
    evaluation.add_argument("--batch-size", type=positive, default=1)
    evaluation.add_argument("--limit", type=positive)
    compare = commands.add_parser("compare", help="Paired example-level bootstrap on saved predictions")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument("--metric", choices=["rouge1", "rouge2", "rougeL"], default="rouge2")
    compare.add_argument("--bootstrap-samples", type=positive, default=2000)
    compare.add_argument("--seed", type=int, default=17)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Resolve command paths relative to invocation; config-internal paths are instead
    # relative to experiment-code so independent cloud/local launches are predictable.
    if args.command == "prepare":
        from gkdproto.data import prepare
        prepare(args)
    elif args.command == "cache-teacher":
        from gkdproto.evaluation import cache_teacher
        cache_teacher(args)
    elif args.command == "train":
        from gkdproto.training import train
        train(args)
    elif args.command == "evaluate":
        from gkdproto.evaluation import evaluate
        evaluate(args)
    else:
        from gkdproto.evaluation import compare
        compare(args)


if __name__ == "__main__":
    main()
