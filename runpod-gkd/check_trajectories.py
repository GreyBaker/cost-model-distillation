"""Run focused trajectory checks with real ML imports and network blocked."""
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
            raise AssertionError("Real ML import blocked: " + fullname)


if __name__ == "__main__":
    blocker = NoML()
    sys.meta_path.insert(0, blocker)
    try:
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network blocked")):
            result = unittest.TextTestRunner().run(
                unittest.defaultTestLoader.loadTestsFromName("test_trajectories"))
    finally:
        sys.meta_path.remove(blocker)
    raise SystemExit(0 if result.wasSuccessful() else 1)
