"""Run external-workflow checks with real ML imports and network blocked."""
from __future__ import annotations

import argparse
import importlib.abc
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["TQDM_DISABLE"] = "1"
preferred = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp/opencode"
if os.name == "nt" and preferred.is_dir():
    tempfile.tempdir = str(preferred)


class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "transformers", "datasets", "numpy", "accelerate", "pyarrow",
            "safetensors", "tokenizers", "rouge_score", "sentencepiece",
        }:
            raise AssertionError("Real ML import blocked by mocked checks: " + fullname)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--quality-override", action="store_true",
                           help="Run only the focused statistical quality-override checks")
    selection.add_argument("--all", action="store_true", help="Run the full regression suite (default)")
    args = parser.parse_args(argv)
    blocker = NoML()
    sys.meta_path.insert(0, blocker)
    try:
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network blocked by mocked checks")):
            if args.quality_override:
                suite = unittest.defaultTestLoader.loadTestsFromName("test_quality_override")
            else:
                suite = unittest.TestSuite([
                    unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py"),
                    unittest.defaultTestLoader.discover(str(ROOT), pattern="check_scheme.py"),
                    unittest.defaultTestLoader.discover(
                        str(ROOT.parent / "runpod-gkd"), pattern="test_eval_throughput.py",
                        top_level_dir=str(ROOT.parent / "runpod-gkd")),
                ])
            result = unittest.TextTestRunner(verbosity=1).run(suite)
    finally:
        sys.meta_path.remove(blocker)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
