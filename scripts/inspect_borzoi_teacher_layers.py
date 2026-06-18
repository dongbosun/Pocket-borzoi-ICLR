#!/usr/bin/env python
"""Inspect official Mini-Borzoi K562 teacher Keras layers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import _path  # noqa: F401

from pocketreg.borzoi.assets import load_assets_config  # noqa: E402
from pocketreg.borzoi.layer_inspection import (  # noqa: E402
    inspect_keras_model,
    write_layer_report,
    write_summary_markdown,
)
from pocketreg.borzoi.targets import parse_targets  # noqa: E402
from pocketreg.borzoi.teacher import BorzoiK562Teacher  # noqa: E402
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.manifest import read_table  # noqa: E402
from pocketreg.data.sequence import one_hot_encode  # noqa: E402

LOGGER = logging.getLogger("inspect_borzoi_teacher_layers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", "--assets-config", dest="assets_config", default="configs/borzoi_assets.local.yaml", type=Path)
    parser.add_argument("--manifest", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet", type=Path)
    parser.add_argument("--fasta", default=None, type=Path)
    parser.add_argument("--out", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_teacher_layers", type=Path)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1], choices=[0, 1])
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-gene-index", type=int, default=0)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _smoke_predict_shape(
    teacher: BorzoiK562Teacher,
    manifest_path: Path,
    fasta_path: Path,
    gene_index: int,
) -> dict:
    rows = read_table(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    row = rows[min(gene_index, len(rows) - 1)]
    fasta = FastaReader(fasta_path)
    seq = fasta.fetch(str(row["chrom"]), int(row["seq_start"]), int(row["seq_end"]), pad=True)
    one_hot = one_hot_encode(seq, channels_first=False)
    pred = teacher.predict(np.expand_dims(one_hot, axis=0), batch_size=1)
    return {
        "example_id": row.get("example_id"),
        "gene_id": row.get("gene_id"),
        "fold": teacher.fold,
        "input_one_hot_shape": list(one_hot.shape),
        "prediction_shape": list(pred.shape),
        "prediction_dtype": str(pred.dtype),
        "prediction_finite": bool(np.isfinite(pred).all()),
        "prediction_min": float(np.nanmin(pred)),
        "prediction_max": float(np.nanmax(pred)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "assets_config": str(args.assets_config),
                    "folds": args.folds,
                    "out": str(args.out),
                    "smoke": not args.no_smoke,
                },
                indent=2,
            )
        )
        return
    assert_compute_context(
        "inspect_borzoi_teacher_layers",
        allow_local=args.allow_local,
        toy=False,
    )
    print_cluster_context()

    assets = load_assets_config(args.assets_config)
    fasta_path = args.fasta or assets.hg38_fasta
    if not args.no_smoke and fasta_path is None:
        raise ValueError("--fasta is required for smoke prediction when assets config lacks hg38_fasta")
    args.out.mkdir(parents=True, exist_ok=True)

    fold_reports: dict[str, dict] = {}
    smoke_shapes: list[dict] = []
    for fold in args.folds:
        LOGGER.info("Loading fold %s for layer inspection", fold)
        teacher = BorzoiK562Teacher(
            assets,
            target_index=args.target_index,
            fold=fold,
            slice_to_target=False,
        )
        report = inspect_keras_model(teacher.seqnn_model.model)
        report["teacher_metadata"] = teacher.metadata_dict()
        report["output_bins"] = teacher.output_num_bins
        report["output_tracks"] = teacher.output_num_tracks
        report["target_names"] = [
            {
                "index": i,
                "identifier": row.get("identifier"),
                "description": row.get("description") or row.get("desc"),
            }
            for i, row in enumerate(parse_targets(assets.k562_targets))
        ]
        fold_name = f"fold{fold}"
        fold_reports[fold_name] = report
        write_layer_report(report, args.out / f"layers_{fold_name}.json")
        if not args.no_smoke and fold == args.folds[0]:
            smoke_shapes.append(
                _smoke_predict_shape(
                    teacher,
                    manifest_path=args.manifest,
                    fasta_path=Path(fasta_path),
                    gene_index=args.smoke_gene_index,
                )
            )

    write_summary_markdown(fold_reports, args.out / "layer_summary.md")
    if smoke_shapes:
        (args.out / "smoke_prediction_shapes.json").write_text(
            json.dumps(smoke_shapes, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"out": str(args.out), "folds": list(fold_reports), "smoke_shapes": smoke_shapes}, indent=2))


if __name__ == "__main__":
    main()
