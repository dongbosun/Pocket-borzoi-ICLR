#!/usr/bin/env python
"""Build a uint8 memmap sequence cache for Pocket-Decima v2 manifests."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.data.fasta import FastaReader
from pocketreg.data.sequence import one_hot_encode
from pocketreg.data.v2 import DecimaV2GeneSequenceDataset
from pocketreg.training.utils import setup_logging

LOGGER = logging.getLogger("build_sequence_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--channels", type=int, default=5, choices=[4, 5])
    parser.add_argument("--dtype", default="uint8", choices=["uint8"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional debug limit; writes a cache for the first N rows.")
    parser.add_argument("--validate-n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def expected_bytes(n_rows: int, channels: int, context_len: int, dtype: str) -> int:
    return int(n_rows) * int(channels) * int(context_len) * np.dtype(dtype).itemsize


def cache_row(reader: FastaReader, row: pd.Series, channels: int) -> np.ndarray:
    chrom = row.get("fasta_chrom", row["chrom"])
    seq = reader.fetch(chrom, int(row["seq_start"]), int(row["seq_end"]), pad=True)
    dna = one_hot_encode(seq).astype(np.uint8)
    if channels == 4:
        return dna
    mask = DecimaV2GeneSequenceDataset._gene_mask(row).astype(np.uint8)
    return np.concatenate([dna, mask[None, :]], axis=0).astype(np.uint8)


def validate_cache(
    reader: FastaReader,
    manifest: pd.DataFrame,
    mmap: np.memmap,
    channels: int,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    take = min(int(n), len(manifest))
    if take <= 0:
        return {"n_checked": 0, "passed": False, "reason": "empty manifest"}
    row_indices = rng.choice(np.arange(len(manifest)), size=take, replace=False)
    max_abs_diff = 0
    failures: list[dict[str, Any]] = []
    for row_pos in row_indices:
        row = manifest.iloc[int(row_pos)]
        expected = cache_row(reader, row, channels)
        observed = np.asarray(mmap[int(row_pos)])
        diff = int(np.max(np.abs(observed.astype(np.int16) - expected.astype(np.int16))))
        max_abs_diff = max(max_abs_diff, diff)
        if diff != 0:
            failures.append({"row_pos": int(row_pos), "gene_id": str(row.get("gene_id")), "max_abs_diff": diff})
            if len(failures) >= 5:
                break
    return {
        "n_checked": int(take),
        "passed": len(failures) == 0,
        "max_abs_diff": int(max_abs_diff),
        "failures": failures,
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    args = parse_args()
    setup_logging()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {args.fasta}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = args.out_dir / f"sequence_cache_{args.dtype}.dat"
    summary_path = args.out_dir / "cache_summary.json"
    index_path = args.out_dir / "cache_index.parquet"
    if (cache_file.exists() or summary_path.exists() or index_path.exists()) and not args.force:
        raise FileExistsError(
            f"Cache outputs already exist in {args.out_dir}; use a new directory or pass --force."
        )

    manifest = pd.read_parquet(args.manifest).reset_index(drop=True)
    if args.limit is not None:
        manifest = manifest.iloc[: int(args.limit)].copy().reset_index(drop=True)
    if manifest.empty:
        raise ValueError("Manifest is empty.")
    context_len = int(args.context_len or manifest["context_len"].iloc[0])
    if not manifest["context_len"].astype(int).eq(context_len).all():
        raise ValueError("All manifest rows must have the same context_len for one cache.")

    n_rows = len(manifest)
    shape = (n_rows, int(args.channels), context_len)
    n_bytes = expected_bytes(n_rows, int(args.channels), context_len, args.dtype)
    LOGGER.info(
        "Building sequence cache: rows=%s channels=%s context=%s expected_size=%.2f GB",
        n_rows,
        args.channels,
        context_len,
        n_bytes / 1024**3,
    )
    reader = FastaReader(args.fasta)
    mmap = np.memmap(cache_file, mode="w+", dtype=np.dtype(args.dtype), shape=shape)
    rows = []
    for row_pos, row in tqdm(list(manifest.iterrows()), desc="sequence-cache"):
        arr = cache_row(reader, row, int(args.channels))
        if arr.shape != (int(args.channels), context_len):
            raise ValueError(
                f"Cache row shape mismatch at row {row_pos}: got {arr.shape}, expected {(args.channels, context_len)}"
            )
        mmap[int(row_pos)] = arr
        rows.append(
            {
                "cache_idx": int(row_pos),
                "gene_idx": int(row["gene_idx"]),
                "gene_id": str(row["gene_id"]),
                "chrom": str(row["chrom"]),
                "fasta_chrom": str(row.get("fasta_chrom", row["chrom"])),
                "seq_start": int(row["seq_start"]),
                "seq_end": int(row["seq_end"]),
                "context_len": int(context_len),
                "split": str(row["split"]),
            }
        )
    mmap.flush()
    cache_index = pd.DataFrame(rows)
    cache_index.to_parquet(index_path, index=False)
    validation = validate_cache(reader, manifest, mmap, int(args.channels), int(args.validate_n), int(args.seed))
    if not validation["passed"]:
        raise RuntimeError(f"Sequence cache validation failed: {validation}")
    summary = {
        "job": "build_sequence_cache",
        "status": "completed",
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "manifest": str(args.manifest.resolve()),
        "fasta": str(args.fasta.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "cache_file": cache_file.name,
        "index_file": index_path.name,
        "n_rows": int(n_rows),
        "shape": [int(x) for x in shape],
        "dtype": args.dtype,
        "channels": int(args.channels),
        "context_len": int(context_len),
        "expected_bytes": int(n_bytes),
        "expected_gb": float(n_bytes / 1024**3),
        "split_counts": {str(k): int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "validation": validation,
    }
    write_json(summary, summary_path)
    LOGGER.info("Saved cache summary: %s", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
