#!/usr/bin/env python
"""Sanity-check Borzoi teacher wrapper against cached ref and delta labels."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import _path  # noqa: F401
import numpy as np
import pandas as pd

from pocketreg.borzoi.assets import load_assets_config  # noqa: E402
from pocketreg.borzoi.output_mapping import BorzoiOutputMapper  # noqa: E402
from pocketreg.borzoi.teacher import BorzoiK562Teacher  # noqa: E402
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.manifest import read_table  # noqa: E402
from pocketreg.data.sequence import apply_snv, one_hot_encode  # noqa: E402

LOGGER = logging.getLogger("borzoi_sanity_check_teacher_wrapper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-config", default="configs/borzoi_assets.local.yaml", type=Path)
    parser.add_argument("--manifest", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet", type=Path)
    parser.add_argument("--fasta", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/external/reference/hg38/hg38.fa", type=Path)
    parser.add_argument("--ref-cache", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_labels.parquet", type=Path)
    parser.add_argument("--rich-cache", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_ref_rich.labels.parquet", type=Path)
    parser.add_argument("--delta-labels", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/interim/teacher_cache/k562_delta_labels.parquet", type=Path)
    parser.add_argument("--out", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/reports/borzoi_sanity", type=Path)
    parser.add_argument("--n-genes", type=int, default=16)
    parser.add_argument("--n-variants", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    return parser.parse_args()


def _mapper(row: dict[str, Any], teacher: BorzoiK562Teacher) -> BorzoiOutputMapper:
    return BorzoiOutputMapper(
        input_seq_start=int(row["seq_start"]),
        input_len=int(row.get("input_len", teacher.input_len)),
        output_num_bins=teacher.output_num_bins,
        bin_size=teacher.bin_size,
        target_index=teacher.mapper_target_index,
        output_core_start=int(row["output_core_start"]) if row.get("output_core_start") is not None else None,
    )


def _aggregate(row: dict[str, Any], output: np.ndarray, teacher: BorzoiK562Teacher) -> float:
    result = _mapper(row, teacher).aggregate_gene_body(
        output,
        int(row["gene_start"]),
        int(row["gene_end"]),
        mode="gene_body_log1p_mean",
    )
    if result is None:
        raise ValueError(f"No gene-body bins for {row.get('example_id')}")
    return float(result.q)


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "max_abs": float("nan"), "mean_abs": float("nan")}
    return {"n": int(arr.size), "max_abs": float(np.max(np.abs(arr))), "mean_abs": float(np.mean(np.abs(arr)))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    assert_compute_context("borzoi_sanity_check_teacher_wrapper", allow_local=args.allow_local, toy=args.toy)
    print_cluster_context()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    manifest = pd.DataFrame(read_table(args.manifest))
    ref_cache = pd.read_parquet(args.ref_cache)
    rich_cache = pd.read_parquet(args.rich_cache) if args.rich_cache.exists() else pd.DataFrame()
    delta_labels = pd.read_parquet(args.delta_labels)
    genes = manifest.merge(ref_cache[["example_id", "q_teacher", "status"]], on="example_id", how="inner")
    genes = genes[genes["status"] == "success"].head(args.n_genes).copy()
    if genes.empty:
        raise ValueError("No genes available for sanity check")
    rich_by_id = rich_cache.set_index("example_id").to_dict(orient="index") if not rich_cache.empty else {}

    assets = load_assets_config(args.assets_config)
    fasta = FastaReader(args.fasta)
    teacher0 = BorzoiK562Teacher(assets, target_index=args.target_index, fold=0)
    teacher1 = BorzoiK562Teacher(assets, target_index=args.target_index, fold=1)

    ref_records: list[dict[str, Any]] = []
    for _, row in genes.iterrows():
        row_dict = row.to_dict()
        seq = fasta.fetch(str(row_dict["chrom"]), int(row_dict["seq_start"]), int(row_dict["seq_end"]), pad=True)
        x = one_hot_encode(seq, channels_first=False)[None, ...]
        pred0 = teacher0.predict(x, batch_size=1)[0]
        pred1 = teacher1.predict(x, batch_size=1)[0]
        q0 = _aggregate(row_dict, pred0, teacher0)
        q1 = _aggregate(row_dict, pred1, teacher1)
        rich = rich_by_id.get(row_dict["example_id"], {})
        ref_records.append(
            {
                "example_id": row_dict["example_id"],
                "q_cache_old": float(row_dict["q_teacher"]),
                "q_recomputed_fold0": q0,
                "q_recomputed_fold1": q1,
                "q_old_diff_fold0": q0 - float(row_dict["q_teacher"]),
                "rich_fold0_diff": q0 - float(rich.get("primary_0_q_fold0", np.nan)),
                "rich_fold1_diff": q1 - float(rich.get("primary_0_q_fold1", np.nan)),
            }
        )

    variant_rows = delta_labels[delta_labels["status"] == "success"].head(args.n_variants).copy()
    variant_rows = variant_rows.merge(manifest, on="example_id", how="left", suffixes=("", "_manifest"))
    delta_records: list[dict[str, Any]] = []
    for _, row in variant_rows.iterrows():
        row_dict = row.to_dict()
        seq = fasta.fetch(str(row_dict["chrom"]), int(row_dict["seq_start"]), int(row_dict["seq_end"]), pad=True)
        x_ref = one_hot_encode(seq, channels_first=False)[None, ...]
        alt_seq = apply_snv(
            seq,
            genomic_pos_0based=int(row_dict["pos_0based"]),
            seq_start_0based=int(row_dict["seq_start"]),
            ref=str(row_dict["ref"]),
            alt=str(row_dict["alt"]),
        ).alt_sequence
        x_alt = one_hot_encode(alt_seq, channels_first=False)[None, ...]
        q_ref = _aggregate(row_dict, teacher0.predict(x_ref, batch_size=1)[0], teacher0)
        q_alt = _aggregate(row_dict, teacher0.predict(x_alt, batch_size=1)[0], teacher0)
        delta = q_alt - q_ref
        delta_records.append(
            {
                "variant_example_id": row_dict["variant_example_id"],
                "delta_cache": float(row_dict["delta_teacher"]),
                "delta_recomputed": delta,
                "delta_diff": delta - float(row_dict["delta_teacher"]),
                "q_ref_cache": float(row_dict["q_ref_teacher"]),
                "q_ref_recomputed": q_ref,
                "q_alt_cache": float(row_dict["q_alt_teacher"]),
                "q_alt_recomputed": q_alt,
            }
        )

    pd.DataFrame(ref_records).to_csv(args.out / "ref_sanity_rows.csv", index=False)
    pd.DataFrame(delta_records).to_csv(args.out / "delta_sanity_rows.csv", index=False)
    summary = {
        "n_genes": int(len(ref_records)),
        "n_variants": int(len(delta_records)),
        "runtime_seconds": time.time() - started,
        "teacher_fold0_shape": {
            "input_len": teacher0.input_len,
            "output_bins": teacher0.output_num_bins,
            "output_tracks": teacher0.output_num_tracks,
            "bin_size": teacher0.bin_size,
        },
        "ref_old_cache_diff_fold0": _summary([r["q_old_diff_fold0"] for r in ref_records]),
        "ref_rich_fold0_diff": _summary([r["rich_fold0_diff"] for r in ref_records]),
        "ref_rich_fold1_diff": _summary([r["rich_fold1_diff"] for r in ref_records]),
        "delta_cache_diff": _summary([r["delta_diff"] for r in delta_records]),
        "outputs": {
            "ref_rows": str(args.out / "ref_sanity_rows.csv"),
            "delta_rows": str(args.out / "delta_sanity_rows.csv"),
        },
    }
    (args.out / "teacher_wrapper_sanity.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
