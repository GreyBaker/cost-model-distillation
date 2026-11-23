"""Read-only reuse of sibling SFT utilities; never patch the SFT package."""
from pathlib import Path
import sys

SFT_CODE = Path(__file__).resolve().parents[2] / "runpod-sft"
if not (SFT_CODE / "sftlib/core.py").is_file():
    raise RuntimeError("Keep runpod-gkd/ alongside the original runpod-sft/ directory")
sys.path.insert(0, str(SFT_CODE))

from sftlib.core import (Cursor, RunLock, atomic_json, check_space, environment,
                         read_json, sha256, sync_directory, sync_tree,
                         tree_manifest, verify_tree)
from sftlib import checkpoints
