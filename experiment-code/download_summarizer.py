"""Plan or explicitly download the pinned jordiclive summarizer; no ML imports.

Default operation only inspects remote file metadata. --download transfers weights;
--verify-only checks an existing completed download without network access.
The original assets/manifest.json and assets/teacher are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from download_assets import atomic_json, download, request, valid

ROOT = Path(__file__).resolve().parent
REPO = "jordiclive/flan-t5-3b-summarizer"
REVISION = "e1f3331b8e0de87602d944da709aa80ece72743f"
WEIGHT_SHA256 = "94ca01edf44c269e06c26bbbc310bc91a17ac94378925dda96dc52b9e2e7af50"
DESTINATION = ROOT / "assets" / "jordiclive_flan_t5_3b_summarizer"
FILES = {"README.md", "config.json", "pytorch_model.bin", "special_tokens_map.json",
         "spiece.model", "tokenizer.json", "tokenizer_config.json"}
MANIFEST = "download_manifest.json"


def selected_files(metadata: dict) -> list[dict]:
    if metadata.get("sha") != REVISION or metadata.get("id") != REPO:
        raise ValueError("Unexpected summarizer repository/revision")
    files = [item for item in metadata["siblings"] if item["rfilename"] in FILES]
    if len(files) != len(FILES) or {item["rfilename"] for item in files} != FILES:
        raise ValueError("Pinned repository lacks the expected model/tokenizer/card files")
    for item in files:
        if not isinstance(item.get("size"), int) or item["size"] < 1:
            raise ValueError("Missing asset byte size")
        if not item.get("lfs", {}).get("sha256") and not item.get("blobId"):
            raise ValueError("Missing asset integrity identity")
    weight = next(item for item in files if item["rfilename"] == "pytorch_model.bin")
    if weight.get("lfs", {}).get("sha256") != WEIGHT_SHA256:
        raise ValueError("Pinned model weight hash differs from the audited artifact")
    return sorted(files, key=lambda item: item["rfilename"])


def remote_metadata() -> dict:
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with request(url) as response:
        return json.load(response)


def verify_download(directory: Path) -> dict:
    """Full byte/hash check, without deserializing model weights."""
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    files = selected_files(manifest)
    if manifest.get("status") != "complete":
        raise ValueError("Summarizer download is incomplete; rerun --download")
    for item in files:
        if not valid(directory / item["rfilename"], item):
            raise ValueError(f"Summarizer integrity failure: {item['rfilename']}")
    return manifest


def plan(metadata: dict, directory: Path) -> dict:
    files = selected_files(metadata)
    return {"repo_id": REPO, "revision": REVISION, "destination": str(directory),
            "weight_sha256": WEIGHT_SHA256, "total_bytes": sum(x["size"] for x in files),
            "files": [{"path": x["rfilename"], "bytes": x["size"]} for x in files],
            "operation": "metadata only; use --download to transfer files"}


def fetch(metadata: dict, directory: Path) -> None:
    files = selected_files(metadata)
    if directory.exists() and any(directory.iterdir()):
        if not (directory / MANIFEST).exists():
            raise FileExistsError(f"Refusing a nonempty directory without this download's manifest: {directory}")
        previous = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        selected_files(previous)
    directory.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for item in files:
        final = directory / item["rfilename"]
        partial = directory / (item["rfilename"] + ".part")
        existing = final if final.exists() else partial
        remaining += max(0, item["size"] - (existing.stat().st_size if existing.exists() else 0))
    if shutil.disk_usage(directory).free < remaining + 1024 ** 3:
        raise OSError("Insufficient free disk for remaining download plus 1 GiB headroom")
    manifest = {"id": REPO, "sha": REVISION, "status": "in_progress", "siblings": files,
                "files": [], "note": "FLAN-T5-XL mixed-dataset summarizer, not the GKD paper teacher"}
    atomic_json(directory / MANIFEST, manifest)
    try:
        for item in files:
            name = item["rfilename"]
            digest = download(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                              directory / name, item)
            manifest["files"].append({"path": name, "bytes": item["size"], "sha256": digest})
            atomic_json(directory / MANIFEST, manifest)
        manifest["status"] = "complete"
        atomic_json(directory / MANIFEST, manifest)
    except Exception as error:
        manifest.update(status="incomplete", error=str(error))
        atomic_json(directory / MANIFEST, manifest)
        raise
    print(f"Summarizer downloaded and checksum-verified: {directory}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if args.verify_only:
        verify_download(directory)
        print(f"Verified all pinned files: {directory}")
        return
    metadata = remote_metadata()
    print(json.dumps(plan(metadata, directory), indent=2))
    if args.download:
        fetch(metadata, directory)


if __name__ == "__main__":
    main()
