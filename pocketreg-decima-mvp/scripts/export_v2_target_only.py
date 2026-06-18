#!/usr/bin/env python
"""Export a Pocket-Decima v2 checkpoint with only target inference outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.models.tcn import build_v2_model, count_parameters, estimate_model_size_mb


LOGGER = logging.getLogger("export_v2_target_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def _target_only_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keep_prefixes = ("stem.", "trunk.", "final_head.")
    return {key: value for key, value in state_dict.items() if key.startswith(keep_prefixes)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    model_config["n_replicates"] = 0
    model_config["n_aux"] = 0
    model_config["n_mid"] = 0
    model_config["n_residual"] = 0

    model = build_v2_model(model_config)
    state_dict = _target_only_state_dict(checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("rep_head.", "aux_head.", "mid_head.", "residual_head."))]
    missing = [key for key in missing if not key.startswith(("rep_head.", "aux_head.", "mid_head.", "residual_head."))]
    if missing or unexpected:
        raise RuntimeError(f"Target-only load mismatch. missing={missing}, unexpected={unexpected}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export = {
        "version": "pocket-decima-targeted-distillation-v2-target-only",
        "source_checkpoint": str(args.checkpoint),
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "context_len": checkpoint.get("context_len"),
        "normalizers": checkpoint.get("normalizers", {}),
        "label_columns": {"final": checkpoint.get("label_columns", {}).get("final", [])},
        "source_epoch": checkpoint.get("epoch"),
        "source_metrics": checkpoint.get("metrics", {}),
        "model_params": count_parameters(model),
        "model_size_mb": estimate_model_size_mb(model),
    }
    torch.save(export, args.out)

    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "out": str(args.out),
                "source_checkpoint": str(args.checkpoint),
                "context_len": export["context_len"],
                "model_config": model_config,
                "model_params": export["model_params"],
                "model_size_mb": export["model_size_mb"],
                "final_label_columns": export["label_columns"]["final"],
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("Exported target-only checkpoint to %s", args.out)
    LOGGER.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
