#!/usr/bin/env python
"""Train a small DNA CNN student on a Decima distillation manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.training.train_loop import train_from_config
from pocketreg.training.utils import load_yaml, setup_logging

LOGGER = logging.getLogger("train_decima_student")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    setup_logging()
    config = load_yaml(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    out_dir = train_from_config(
        config,
        run_name=args.run_name,
        manifest_path=args.manifest,
        fasta_path=args.fasta,
    )
    LOGGER.info("Training complete. Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
