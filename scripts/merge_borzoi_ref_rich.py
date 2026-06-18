#!/usr/bin/env python
"""Merge per-shard rich Mini-Borzoi reference caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401

from pocketreg.borzoi.rich_teacher_cache import summarize_rich_labels, write_summary  # noqa: E402
from pocketreg.data.manifest import read_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/ref_rich_shards", type=Path)
    parser.add_argument("--out-prefix", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich", type=Path)
    parser.add_argument("--manifest", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_out = args.out_prefix.with_suffix(".labels.parquet")
    index_out = args.out_prefix.with_suffix(".index.parquet")
    profile_out = args.out_prefix.with_suffix(".profiles.npz")
    summary_out = args.out_prefix.with_suffix(".summary.json")
    if label_out.exists() and profile_out.exists() and not args.overwrite:
        raise FileExistsError(f"Outputs exist; use --overwrite: {label_out}, {profile_out}")

    label_files = sorted(args.cache_dir.glob("*.labels.parquet"))
    if not label_files:
        raise FileNotFoundError(f"No *.labels.parquet files in {args.cache_dir}")

    frames = [pd.read_parquet(path) for path in label_files]
    labels = pd.concat(frames, ignore_index=True)
    if labels["example_id"].duplicated().any():
        dup = labels.loc[labels["example_id"].duplicated(), "example_id"].head().tolist()
        raise ValueError(f"Duplicate example_id in rich cache shards: {dup}")
    manifest = pd.DataFrame(read_table(args.manifest))
    missing = sorted(set(manifest["example_id"]) - set(labels["example_id"]))
    labels = labels.merge(
        manifest[["example_id", "split"]].rename(columns={"split": "manifest_split"}),
        on="example_id",
        how="left",
    )
    labels["profile_row"] = np.arange(len(labels), dtype=np.int64)

    profile_arrays: dict[str, list[np.ndarray]] = {}
    for label_path in label_files:
        npz_path = label_path.with_suffix("").with_suffix(".profiles.npz")
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        for key in data.files:
            profile_arrays.setdefault(key, []).append(data[key])
    merged_arrays = {key: np.concatenate(parts, axis=0) for key, parts in profile_arrays.items()}
    for key, arr in merged_arrays.items():
        if arr.shape[0] != len(labels):
            raise ValueError(f"Profile array {key} rows {arr.shape[0]} != labels {len(labels)}")

    label_out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(label_out, index=False)
    index = labels[["profile_row", "example_id", "gene_id", "split", "status"]].copy()
    index.to_parquet(index_out, index=False)
    tmp = profile_out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **merged_arrays)
    tmp.replace(profile_out)
    summary = {
        "task": "merge_borzoi_ref_rich",
        "cache_dir": str(args.cache_dir),
        "num_shards": len(label_files),
        "label_out": str(label_out),
        "index_out": str(index_out),
        "profile_out": str(profile_out),
        "profile_shapes": {key: list(arr.shape) for key, arr in merged_arrays.items()},
        "missing_manifest_examples": len(missing),
        "split_counts": labels["split"].value_counts(dropna=False).to_dict() if "split" in labels else {},
        **summarize_rich_labels(labels.to_dict(orient="records")),
    }
    write_summary(summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
