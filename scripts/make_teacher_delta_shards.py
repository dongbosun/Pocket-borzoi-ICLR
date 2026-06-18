#!/usr/bin/env python
"""Shard variant-gene rows for Borzoi delta teacher inference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _path  # noqa: F401

from pocketreg.data.manifest import read_table, write_table  # noqa: E402
from pocketreg.data.shard import padded_shard_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--shard-by", choices=["gene", "row"], default="gene")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    rows = read_table(args.variants)
    if not rows:
        raise ValueError(f"No variants in {args.variants}")

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    if args.shard_by == "row":
        for i, row in enumerate(rows):
            shards[i % args.num_shards].append(row)
    else:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[str(row["example_id"])].append(row)
        for i, key in enumerate(sorted(groups)):
            shards[i % args.num_shards].extend(groups[key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_info = []
    for shard_id, shard_rows in enumerate(shards):
        path = args.out_dir / padded_shard_name(shard_id)
        example_ids = {row.get("example_id") for row in shard_rows}
        shard_info.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "num_rows": len(shard_rows),
                "num_genes": len(example_ids),
            }
        )
        if args.dry_run:
            continue
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Shard exists: {path}; pass --overwrite")
        write_table(shard_rows, path)

    summary = {
        "variants": str(args.variants),
        "out_dir": str(args.out_dir),
        "num_input_rows": len(rows),
        "num_shards": args.num_shards,
        "shard_by": args.shard_by,
        "min_rows": min(len(s) for s in shards),
        "max_rows": max(len(s) for s in shards),
        "shards": shard_info,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}, indent=2))
    if not args.dry_run:
        (args.out_dir / "shards_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
