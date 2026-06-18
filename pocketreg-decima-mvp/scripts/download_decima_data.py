#!/usr/bin/env python
"""Download Decima metadata.h5ad from Hugging Face."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.training.utils import setup_logging

LOGGER = logging.getLogger("download_decima_data")


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    setup_logging()

    if args.out.exists() and not args.force:
        LOGGER.info("File already exists: %s (%.3f GiB)", args.out, file_size_gb(args.out))
        return
    if args.no_download:
        if not args.out.exists():
            raise FileNotFoundError(f"--no-download was set, but file does not exist: {args.out}")
        LOGGER.info("Found existing file: %s (%.3f GiB)", args.out, file_size_gb(args.out))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(args.out.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id="Genentech/decima-data",
                repo_type="dataset",
                filename="metadata.h5ad",
            )
        )
        shutil.copy2(downloaded, tmp_out)
        tmp_out.replace(args.out)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted while downloading/copying Decima metadata.")
        if tmp_out.exists():
            tmp_out.unlink()
        raise
    LOGGER.info("Saved %s (%.3f GiB)", args.out, file_size_gb(args.out))


if __name__ == "__main__":
    main()
