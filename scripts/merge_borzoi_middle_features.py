#!/usr/bin/env python
"""Merge per-shard pooled Borzoi middle feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401

from pocketreg.borzoi.middle_features import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/middle_feature_shards", type=Path)
    parser.add_argument("--out-prefix", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_middle_pooled", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_out = args.out_prefix.with_suffix(".index.parquet")
    features_out = args.out_prefix.with_suffix(".features.npz")
    summary_out = args.out_prefix.with_suffix(".summary.json")
    if index_out.exists() and features_out.exists() and not args.overwrite:
        raise FileExistsError(f"Outputs exist; use --overwrite: {args.out_prefix}")
    index_files = sorted(args.cache_dir.glob("*.index.parquet"))
    if not index_files:
        raise FileNotFoundError(f"No *.index.parquet in {args.cache_dir}")
    indices = [pd.read_parquet(path) for path in index_files]
    index = pd.concat(indices, ignore_index=True)
    if index["example_id"].duplicated().any():
        dup = index.loc[index["example_id"].duplicated(), "example_id"].head().tolist()
        raise ValueError(f"Duplicate example_id in middle feature shards: {dup}")
    index["feature_row"] = np.arange(len(index), dtype=np.int64)
    arrays: dict[str, list[np.ndarray]] = {}
    for index_path in index_files:
        npz_path = index_path.with_suffix("").with_suffix(".features.npz")
        data = np.load(npz_path)
        for key in data.files:
            arrays.setdefault(key, []).append(data[key])
    merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    for key, arr in merged.items():
        if arr.shape[0] != len(index):
            raise ValueError(f"{key} rows {arr.shape[0]} != index rows {len(index)}")
    index_out.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_out, index=False)
    tmp = features_out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **merged)
    tmp.replace(features_out)
    summary = {
        "task": "merge_borzoi_middle_features",
        "cache_dir": str(args.cache_dir),
        "num_shards": len(index_files),
        "rows": int(len(index)),
        "index_out": str(index_out),
        "features_out": str(features_out),
        "feature_shapes": {key: list(arr.shape) for key, arr in merged.items()},
        "finite_fraction": {key: float(np.isfinite(arr).mean()) for key, arr in merged.items()},
    }
    write_json(summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
