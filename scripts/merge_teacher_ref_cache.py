#!/usr/bin/env python
"""Merge per-shard reference teacher cache tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import _path  # noqa: F401

from pocketreg.data.manifest import read_table, write_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["example_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_distribution(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    values = [float(row["q_teacher"]) for row in rows if row.get("status") == "success"]
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=50, color="#4C78A8", alpha=0.85)
    plt.xlabel("q_teacher")
    plt.ylabel("genes")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.cache_dir.glob("shard_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.parquet files found in {args.cache_dir}")
    manifest_rows = read_table(args.manifest)
    manifest_by_id = {row["example_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Manifest contains duplicated example_id values")

    cache_rows: list[dict] = []
    for path in shard_paths:
        cache_rows.extend(read_table(path))
    counts = Counter(row.get("example_id") for row in cache_rows)
    duplicates = [example_id for example_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate teacher cache example_id values, first={duplicates[:5]}")

    merged: list[dict] = []
    cache_by_id = {row.get("example_id"): row for row in cache_rows}
    missing_or_failed: list[dict] = []
    for manifest_row in manifest_rows:
        example_id = manifest_row["example_id"]
        cache_row = cache_by_id.get(example_id)
        if cache_row is None:
            missing_or_failed.append(
                {
                    "example_id": example_id,
                    "gene_id": manifest_row.get("gene_id"),
                    "split": manifest_row.get("split"),
                    "status": "missing",
                    "error_message": "",
                }
            )
            continue
        merged_row = {**manifest_row, **cache_row}
        merged.append(merged_row)
        if cache_row.get("status") != "success":
            missing_or_failed.append(
                {
                    "example_id": example_id,
                    "gene_id": manifest_row.get("gene_id"),
                    "split": manifest_row.get("split"),
                    "status": cache_row.get("status"),
                    "error_message": cache_row.get("error_message", ""),
                }
            )

    if args.require_complete and missing_or_failed:
        raise RuntimeError(f"{len(missing_or_failed)} manifest rows missing/failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_table(merged, args.out)
    failed_path = args.out.with_name("missing_or_failed_examples.csv")
    _write_csv(missing_or_failed, failed_path)

    success_rows = [row for row in merged if row.get("status") == "success"]
    values = [float(row["q_teacher"]) for row in success_rows]
    split_counts = Counter(row.get("split", "unknown") for row in success_rows)
    summary = {
        "cache_dir": str(args.cache_dir),
        "out": str(args.out),
        "num_shard_files": len(shard_paths),
        "manifest_rows": len(manifest_rows),
        "merged_rows": len(merged),
        "success_rows": len(success_rows),
        "missing_or_failed_rows": len(missing_or_failed),
        "split_success_counts": dict(split_counts),
        "q_teacher": {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        },
    }
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _plot_distribution(success_rows, args.out.with_name(args.out.stem + "_distribution.png"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
