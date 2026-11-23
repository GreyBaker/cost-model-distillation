"""Download pinned public assets using only the Python standard library.

Does not import torch, deserialize checkpoints, or load dataset tables into RAM.
Model repositories currently provide PyTorch .bin weights, not safetensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
ASSETS = [
    ("model", "google/t5-small-lm-adapt", "4ff52753a4ddd906de166c573db99280719b22f9", "student"),
    ("model", "google/t5-xl-lm-adapt", "f9843a55f0bdc37c957ddba20be0be4870b0a994", "teacher"),
    ("dataset", "EdinburghNLP/xsum", "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9", "xsum"),
]
MODEL_FILES = {
    "README.md", "config.json", "generation_config.json", "pytorch_model.bin",
    "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json",
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def request(url: str, headers: dict | None = None):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "t5-gkd-research-prototype/1.0", **(headers or {})}
    ), timeout=120)


def digest_file(path: Path, algorithm: str = "sha256", git_blob: bool = False) -> str:
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, item: dict) -> bool:
    if not path.exists() or path.stat().st_size != item["size"]:
        return False
    if item.get("lfs", {}).get("sha256"):
        return digest_file(path) == item["lfs"]["sha256"]
    return digest_file(path, "sha1", git_blob=True) == item["blobId"]


def download(url: str, destination: Path, item: dict) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid(destination, item):
        print(f"Verified existing: {destination}", flush=True)
        return item.get("lfs", {}).get("sha256") or digest_file(destination)
    if destination.exists():
        raise RuntimeError(f"Existing final file failed integrity check; investigate before replacing: {destination}")
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, 4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item["size"]:
                raise RuntimeError(f"Oversized partial file; inspect it manually: {partial}")
            if offset < item["size"]:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                # A fresh resolve request avoids reusing an expired CDN redirect.
                with request(url + "?download=true&request_id=" + uuid.uuid4().hex, headers) as response:
                    status = response.status
                    if offset and status == 206:
                        expected = f"bytes {offset}-"
                        if not response.headers.get("Content-Range", "").startswith(expected):
                            raise RuntimeError("Server returned an unexpected byte range")
                    elif offset and status == 200:
                        offset = 0  # Server ignored Range: restart this partial, not a final file.
                    elif status not in (200, 206):
                        raise RuntimeError(f"Unexpected download status {status}")
                    print(f"Downloading {destination.name}: {offset:,}/{item['size']:,} bytes", flush=True)
                    reported = offset
                    with partial.open("ab" if offset else "wb") as stream:
                        while chunk := response.read(4 * 1024 * 1024):
                            stream.write(chunk)
                            offset += len(chunk)
                            if offset - reported >= 512 * 1024 * 1024:
                                print(f"  {destination.name}: {offset:,}/{item['size']:,}", flush=True)
                                reported = offset
            if not valid(partial, item):
                raise ValueError(f"Size/hash mismatch for {partial}; partial retained for inspection")
            sha256 = item.get("lfs", {}).get("sha256") or digest_file(partial)
            os.replace(partial, destination)
            print(f"Verified download: {destination.name}; sha256={sha256}", flush=True)
            return sha256
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            print(f"Attempt {attempt}/3 failed: {error}", flush=True)
            if attempt == 3:
                raise
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    base = args.assets_dir.resolve()
    manifest = {"status": "in_progress", "note": "Pretrained LM-adapted checkpoints; not XSum-SFT models", "assets": {}}
    manifest_path = base / "manifest.json"
    atomic_json(manifest_path, manifest)
    try:
        for kind, repo, revision, name in ASSETS:
            plural = "models" if kind == "model" else "datasets"
            with request(f"https://huggingface.co/api/{plural}/{repo}/revision/{revision}?blobs=true") as response:
                metadata = json.load(response)
            if metadata["sha"] != revision:
                raise RuntimeError(f"Unexpected revision for {repo}")
            entry = {"repo_id": repo, "revision": revision, "path": name, "files": [], "complete": False}
            manifest["assets"][name] = entry
            atomic_json(manifest_path, manifest)
            for item in metadata["siblings"]:
                filename = item["rfilename"]
                wanted = filename in MODEL_FILES if kind == "model" else (filename == "README.md" or (filename.startswith("data/") and filename.endswith(".parquet")))
                if not wanted:
                    continue
                prefix = "" if kind == "model" else "datasets/"
                url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{filename}"
                sha256 = download(url, base / name / filename, item)
                entry["files"].append({"path": filename, "bytes": item["size"], "sha256": sha256})
                atomic_json(manifest_path, manifest)
            if kind == "model" and not any(f["path"] == "pytorch_model.bin" for f in entry["files"]):
                raise RuntimeError(f"Missing model weights: {repo}")
            if kind == "dataset" and sum(f["path"].endswith(".parquet") for f in entry["files"]) != 3:
                raise RuntimeError("Expected three XSum parquet split files")
            entry["complete"] = True
            atomic_json(manifest_path, manifest)
        manifest["status"] = "complete"
        atomic_json(manifest_path, manifest)
        print(f"All assets downloaded and checksum-verified: {manifest_path}", flush=True)
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error)
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
