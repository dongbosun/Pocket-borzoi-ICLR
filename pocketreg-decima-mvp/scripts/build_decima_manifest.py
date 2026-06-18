#!/usr/bin/env python
"""Build a gene-centered manifest for one selected Decima pseudobulk."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocketreg.data.anndata_utils import select_pseudobulk
from pocketreg.data.manifest import build_manifest_from_anndata, save_manifest_outputs
from pocketreg.training.utils import setup_logging

LOGGER = logging.getLogger("build_decima_manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--context-len", type=int, default=65536)
    parser.add_argument("--label-layer", default="preds")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--target-query")
    parser.add_argument("--organ")
    parser.add_argument("--tissue")
    parser.add_argument("--disease")
    parser.add_argument("--cell-type-contains")
    parser.add_argument("--region-contains")
    parser.add_argument("--subregion-contains")
    parser.add_argument("--celltype-coarse-contains")
    parser.add_argument("--split-mode", choices=["chromosome", "decima_dataset", "fold"], default="chromosome")
    parser.add_argument("--include-sex-chromosomes", action="store_true")
    parser.add_argument("--all-gene-types", action="store_true")
    parser.add_argument("--max-frac-n", type=float, default=0.05)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--coordinate-convention", choices=["zero_based", "one_based"], default="zero_based")
    parser.add_argument("--skip-fasta-check", action="store_true")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--test-fold", type=int, default=1)
    args = parser.parse_args()
    setup_logging()

    if not args.adata.exists():
        raise FileNotFoundError(f"AnnData file not found: {args.adata}")
    if args.fasta is None and not args.skip_fasta_check:
        raise ValueError("Provide --fasta or pass --skip-fasta-check for dry manifest mode.")
    adata = ad.read_h5ad(args.adata)
    target_idx, target_metadata = select_pseudobulk(
        adata,
        target_index=args.target_index,
        query=args.target_query,
        organ=args.organ,
        tissue=args.tissue,
        disease=args.disease,
        cell_type_contains=args.cell_type_contains,
        region_contains=args.region_contains,
        subregion_contains=args.subregion_contains,
        celltype_coarse_contains=args.celltype_coarse_contains,
    )
    manifest, target_metadata, summary = build_manifest_from_anndata(
        adata,
        target_idx,
        args.label_layer,
        args.context_len,
        split_mode=args.split_mode,
        include_sex_chromosomes=args.include_sex_chromosomes,
        all_gene_types=args.all_gene_types,
        max_frac_n=args.max_frac_n,
        max_genes=args.max_genes,
        seed=args.seed,
        coordinate_convention=args.coordinate_convention,
        fasta_path=args.fasta,
        skip_fasta_check=args.skip_fasta_check,
        val_fold=args.val_fold,
        test_fold=args.test_fold,
    )
    save_manifest_outputs(manifest, args.out, target_metadata, summary)
    LOGGER.info("Saved manifest to %s with %s genes", args.out, len(manifest))
    LOGGER.info("Split counts: %s", summary["split_counts"])


if __name__ == "__main__":
    main()
