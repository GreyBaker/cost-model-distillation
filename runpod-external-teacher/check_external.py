"""Run all external-workflow checks with real ML imports and network blocked."""
from __future__ import annotations

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


def main():
    sys.meta_path.insert(0, NoML())
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py"),
        unittest.defaultTestLoader.discover(str(ROOT), pattern="check_scheme.py"),
    ])
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network blocked by mocked checks")):
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
