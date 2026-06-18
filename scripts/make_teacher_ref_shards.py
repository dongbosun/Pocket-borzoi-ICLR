#!/usr/bin/env python
"""Split a gene manifest into teacher inference shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from pocketreg.data.manifest import read_table, write_table  # noqa: E402
from pocketreg.data.shard import assign_even_shards, padded_shard_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_table(args.manifest)
    if not rows:
        raise ValueError(f"Manifest has no rows: {args.manifest}")
    shards = assign_even_shards(rows, args.num_shards)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for shard_id, shard_rows in enumerate(shards):
        path = args.out_dir / padded_shard_name(shard_id)
        manifest_rows.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "num_rows": len(shard_rows),
                "first_example_id": shard_rows[0].get("example_id") if shard_rows else None,
                "last_example_id": shard_rows[-1].get("example_id") if shard_rows else None,
            }
        )
        if args.dry_run:
            continue
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Shard exists: {path}; pass --overwrite")
        write_table(shard_rows, path)

    summary = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "num_input_rows": len(rows),
        "num_shards": args.num_shards,
        "min_shard_rows": min(len(s) for s in shards),
        "max_shard_rows": max(len(s) for s in shards),
        "shards": manifest_rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}, indent=2))
    if not args.dry_run:
        (args.out_dir / "shards_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
