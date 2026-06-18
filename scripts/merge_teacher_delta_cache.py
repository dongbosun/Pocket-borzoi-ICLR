#!/usr/bin/env python
"""Merge per-shard Borzoi delta teacher cache tables."""

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
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["variant_example_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], out_prefix: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    success = [row for row in rows if row.get("status") == "success"]
    if not success:
        return
    deltas = [float(row["delta_teacher"]) for row in success]
    abs_deltas = [abs(v) for v in deltas]
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(deltas, bins=80, color="#4C78A8", alpha=0.85)
    plt.xlabel("delta_teacher")
    plt.ylabel("variants")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_delta_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist([v for v in abs_deltas if v > 0], bins=80, log=True, color="#F58518", alpha=0.85)
    plt.xlabel("|delta_teacher|")
    plt.ylabel("variants")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_abs_delta_distribution.png"), dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.cache_dir.glob("shard_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.parquet files found in {args.cache_dir}")
    variants = read_table(args.variants)
    variant_by_id = {row["variant_example_id"]: row for row in variants}
    if len(variant_by_id) != len(variants):
        raise ValueError("Variant manifest contains duplicate variant_example_id values")
    cache_rows = []
    for path in shard_paths:
        cache_rows.extend(read_table(path))
    counts = Counter(row.get("variant_example_id") for row in cache_rows)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate delta cache variant_example_id values, first={duplicates[:5]}")
    cache_by_id = {row.get("variant_example_id"): row for row in cache_rows}

    merged: list[dict] = []
    failed: list[dict] = []
    for variant in variants:
        variant_id = variant["variant_example_id"]
        cache = cache_by_id.get(variant_id)
        if cache is None:
            failed.append(
                {
                    "variant_example_id": variant_id,
                    "example_id": variant.get("example_id"),
                    "status": "missing",
                    "error_message": "",
                }
            )
            continue
        row = {**variant, **cache}
        merged.append(row)
        if cache.get("status") != "success":
            failed.append(
                {
                    "variant_example_id": variant_id,
                    "example_id": variant.get("example_id"),
                    "status": cache.get("status"),
                    "error_message": cache.get("error_message", ""),
                }
            )

    if args.require_complete and failed:
        raise RuntimeError(f"{len(failed)} variants missing/failed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_table(merged, args.out)
    _write_csv(failed, args.out.with_name("failed_variants.csv"))
    success = [row for row in merged if row.get("status") == "success"]
    abs_deltas = [abs(float(row["delta_teacher"])) for row in success]
    summary = {
        "cache_dir": str(args.cache_dir),
        "variants": str(args.variants),
        "out": str(args.out),
        "num_shard_files": len(shard_paths),
        "variant_manifest_rows": len(variants),
        "merged_rows": len(merged),
        "success_rows": len(success),
        "missing_or_failed_rows": len(failed),
        "split_success_counts": dict(Counter(row.get("split", "unknown") for row in success)),
        "abs_delta": {
            "max": max(abs_deltas) if abs_deltas else None,
            "mean": sum(abs_deltas) / len(abs_deltas) if abs_deltas else None,
            "lt_1e-6": sum(v < 1e-6 for v in abs_deltas) / max(1, len(abs_deltas)),
            "lt_1e-4": sum(v < 1e-4 for v in abs_deltas) / max(1, len(abs_deltas)),
            "lt_1e-3": sum(v < 1e-3 for v in abs_deltas) / max(1, len(abs_deltas)),
        },
    }
    args.out.with_name(args.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _plot(success, args.out.with_name(args.out.stem))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
