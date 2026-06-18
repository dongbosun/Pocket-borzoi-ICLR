#!/usr/bin/env python
"""Submit an sbatch script and record the command/job id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from pocketreg.cluster.slurm import build_sbatch_command, submit_sbatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--array")
    parser.add_argument("--export")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submissions-dir", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/logs/submissions")
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()
    cmd = build_sbatch_command(Path(args.job), array=args.array, export=args.export, extra_args=args.extra)
    print(" ".join(cmd))
    record = submit_sbatch(cmd, dry_run=args.dry_run, submissions_dir=Path(args.submissions_dir))
    print(json.dumps(record.__dict__, indent=2))


if __name__ == "__main__":
    main()
