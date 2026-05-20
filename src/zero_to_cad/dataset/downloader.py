"""Download Zero-To-CAD-1m test parquet shards from Hugging Face."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from zero_to_cad.config import DATASET_REPO, DATASET_TEST_GLOB, dataset_test_dir


ProgressCallback = Callable[[str, int, int], None] | None


def list_remote_test_shards() -> list[str]:
    """List parquet paths under data/test/ in the dataset repo."""
    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    prefix = "data/test/"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    return shards


def download_test_split(
    dest_dir: Path | None = None,
    progress: ProgressCallback = None,
) -> Path:
    """
    Download all test-split parquet shards into dest_dir (default: XDG cache).

    Returns the directory containing the parquet files.
    """
    dest = dest_dir or dataset_test_dir()
    dest.mkdir(parents=True, exist_ok=True)

    shards = list_remote_test_shards()
    if not shards:
        raise RuntimeError(f"No parquet shards found under {DATASET_TEST_GLOB}")

    for i, shard_path in enumerate(tqdm(shards, desc="Downloading test split")):
        if progress:
            progress(shard_path, i, len(shards))

        local = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=shard_path,
            repo_type="dataset",
        )
        target = dest / Path(shard_path).name
        if not target.exists() or target.stat().st_size != Path(local).stat().st_size:
            shutil.copy2(local, target)

    return dest


def is_test_split_downloaded(dest_dir: Path | None = None) -> bool:
    dest = dest_dir or dataset_test_dir()
    return dest.exists() and any(dest.glob("*.parquet"))
