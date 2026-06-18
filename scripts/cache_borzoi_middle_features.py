#!/usr/bin/env python
"""Cache compact pooled Borzoi teacher middle/head-input features."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401

from pocketreg.borzoi.assets import load_assets_config  # noqa: E402
from pocketreg.borzoi.middle_features import pooled_spatial_features, select_layer_name, write_json  # noqa: E402
from pocketreg.borzoi.teacher import BorzoiK562Teacher  # noqa: E402
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.manifest import atomic_write_table, read_table  # noqa: E402
from pocketreg.data.sequence import one_hot_encode  # noqa: E402

LOGGER = logging.getLogger("cache_borzoi_middle_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", default="configs/borzoi_assets.local.yaml", type=Path)
    parser.add_argument("--manifest", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet", type=Path)
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--fasta", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa", type=Path)
    parser.add_argument("--layer-report-dir", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_teacher_layers", type=Path)
    parser.add_argument("--layer-name", default="auto")
    parser.add_argument("--folds", nargs="+", type=int, default=[0], choices=[0, 1])
    parser.add_argument("--out-prefix", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_middle", type=Path)
    parser.add_argument("--n", type=int)
    parser.add_argument("--center-bins", type=int, default=256)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _intermediate_model(teacher: BorzoiK562Teacher, layer_name: str):
    tf = teacher.tf
    layer = teacher.seqnn_model.model.get_layer(layer_name)
    return tf.keras.Model(inputs=teacher.seqnn_model.model.inputs, outputs=layer.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return
    assert_compute_context("cache_borzoi_middle_features", allow_local=args.allow_local, toy=False)
    print_cluster_context()
    index_path = args.out_prefix.with_suffix(".index.parquet")
    features_path = args.out_prefix.with_suffix(".features.npz")
    summary_path = args.out_prefix.with_suffix(".summary.json")
    if index_path.exists() and features_path.exists() and not args.overwrite:
        LOGGER.info("Outputs exist and --overwrite not set; skipping %s", args.out_prefix)
        return

    started = time.time()
    assets = load_assets_config(args.assets_config)
    rows = read_table(args.shard or args.manifest)
    if args.n is not None:
        rows = rows[: args.n]
    fasta = FastaReader(args.fasta)
    index_rows = []
    fold_arrays: dict[str, np.ndarray] = {}
    selected_layers: dict[str, str] = {}

    for fold in args.folds:
        report = args.layer_report_dir / f"layers_fold{fold}.json"
        layer_name = select_layer_name(report, args.layer_name)
        selected_layers[f"fold{fold}"] = layer_name
        LOGGER.info("Loading fold %s and extracting layer %s", fold, layer_name)
        teacher = BorzoiK562Teacher(assets, fold=fold, slice_to_target=False)
        model = _intermediate_model(teacher, layer_name)
        feats = []
        for i, row in enumerate(rows):
            seq = fasta.fetch(str(row["chrom"]), int(row["seq_start"]), int(row["seq_end"]), pad=True)
            if len(seq) != teacher.input_len:
                raise ValueError(f"Sequence length {len(seq)} != teacher input {teacher.input_len}")
            pred = model.predict(np.expand_dims(one_hot_encode(seq, channels_first=False), axis=0), verbose=0)
            feats.append(pooled_spatial_features(pred, center_bins=args.center_bins))
            if fold == args.folds[0]:
                index_rows.append(
                    {
                        "feature_row": i,
                        "example_id": row.get("example_id"),
                        "gene_id": row.get("gene_id"),
                        "split": row.get("split"),
                        "status": "success",
                    }
                )
        fold_arrays[f"features_fold{fold}"] = np.stack(feats, axis=0).astype(np.float16)

    present = [arr.astype(np.float32) for arr in fold_arrays.values()]
    if present:
        fold_arrays["features_mean"] = np.mean(np.stack(present, axis=0), axis=0).astype(np.float16)
        if len(present) > 1:
            fold_arrays["features_std"] = np.std(np.stack(present, axis=0), axis=0).astype(np.float16)

    atomic_write_table(index_rows, index_path)
    tmp = features_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **fold_arrays)
    tmp.replace(features_path)
    finite = {key: float(np.isfinite(arr).mean()) for key, arr in fold_arrays.items()}
    summary = {
        "task": "cache_borzoi_middle_features",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "input": str(args.shard or args.manifest),
        "index_path": str(index_path),
        "features_path": str(features_path),
        "rows": len(index_rows),
        "folds": args.folds,
        "selected_layers": selected_layers,
        "feature_shapes": {key: list(arr.shape) for key, arr in fold_arrays.items()},
        "finite_fraction": finite,
        "runtime_seconds": time.time() - started,
        "examples_per_second": len(index_rows) / max(1e-6, time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
