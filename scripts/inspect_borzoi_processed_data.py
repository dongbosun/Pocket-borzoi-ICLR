#!/usr/bin/env python
"""Inspect official processed K562/Borzoi data directories."""

from __future__ import annotations

import argparse
import json

import _path  # noqa: F401

from pocketreg.borzoi.processed_data import inspect_processed_data_dir, write_processed_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/processed_k562_inspection")
    parser.add_argument("--max-files", type=int, default=5000)
    args = parser.parse_args()
    report = inspect_processed_data_dir(args.data_dir, max_files=args.max_files)
    write_processed_report(report, args.out)
    print(json.dumps({"out": args.out, "looks_usable_as_labels": report["looks_usable_as_labels"]}, indent=2))


if __name__ == "__main__":
    main()
