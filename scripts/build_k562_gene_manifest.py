#!/usr/bin/env python
"""Build a K562 selected-track gene manifest from GTF/FASTA and Borzoi metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from pocketreg.borzoi.assets import detect_shape_metadata, load_assets_config, load_params
from pocketreg.borzoi.targets import find_k562_rnaseq_candidates, parse_targets, row_target_index
from pocketreg.data.fasta import FastaReader
from pocketreg.data.genes import build_gene_manifest_rows, manifest_summary, parse_genes_from_gtf
from pocketreg.data.manifest import write_rows_csv_gz, write_table


def int_or_auto(value: str) -> int | str:
    return "auto" if value == "auto" else int(value)


def require_int(name: str, value) -> int:
    if value in (None, "auto"):
        raise SystemExit(f"Could not auto-detect {name}. Pass --{name.replace('_', '-')} explicitly.")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-config", required=True)
    parser.add_argument("--fasta")
    parser.add_argument("--gtf")
    parser.add_argument("--out", default="/extra/zhanglab0/INDV/dongbos/Pocket-borzoi-ICLR/dataset/manifests/k562_gene_manifest.parquet")
    parser.add_argument("--input-len", default="auto", type=int_or_auto)
    parser.add_argument("--output-num-bins", default="auto", type=int_or_auto)
    parser.add_argument("--bin-size", default="auto", type=int_or_auto)
    parser.add_argument("--target-index", default="auto")
    parser.add_argument("--aggregation", default="gene_body_log1p_mean")
    parser.add_argument("--max-genes", type=int, default=5000)
    parser.add_argument("--min-gene-overlap-fraction", type=float, default=0.8)
    parser.add_argument("--include-non-autosomes", action="store_true")
    parser.add_argument("--include-non-protein-coding", action="store_true")
    args = parser.parse_args()

    config = load_assets_config(args.assets_config)
    fasta_path = Path(args.fasta or config.hg38_fasta or "")
    gtf_path = Path(args.gtf or config.gencode_gtf or "")
    if not fasta_path.exists():
        raise SystemExit(f"FASTA not found: {fasta_path}")
    if not gtf_path.exists():
        raise SystemExit(f"GTF not found: {gtf_path}")

    params = load_params(config.k562_params) if config.k562_params and config.k562_params.exists() else {}
    detected = detect_shape_metadata(params)
    input_len = require_int("input_len", detected["input_len"] if args.input_len == "auto" else args.input_len)
    output_num_bins = require_int(
        "output_num_bins",
        detected["output_num_bins"] if args.output_num_bins == "auto" else args.output_num_bins,
    )
    bin_size = require_int("bin_size", detected["bin_size"] if args.bin_size == "auto" else args.bin_size)

    target_identifier = ""
    target_description = ""
    if args.target_index == "auto":
        if not config.k562_targets or not config.k562_targets.exists():
            raise SystemExit("Cannot auto-detect target index without k562_targets.")
        candidates = find_k562_rnaseq_candidates(parse_targets(config.k562_targets))
        if not candidates:
            raise SystemExit("No K562 RNA-seq target candidates found. Pass --target-index.")
        selected = candidates[0]
        target_index = row_target_index(selected)
        if target_index is None:
            raise SystemExit("Could not infer target index from selected target row. Pass --target-index.")
        target_identifier = selected.get("identifier", selected.get("index", ""))
        target_description = selected.get("description", "")
    else:
        target_index = int(args.target_index)

    fasta = FastaReader(fasta_path)
    genes = parse_genes_from_gtf(
        gtf_path,
        autosomes_only=not args.include_non_autosomes,
        protein_coding_only=not args.include_non_protein_coding,
    )
    rows = build_gene_manifest_rows(
        genes=genes,
        fasta=fasta,
        input_len=input_len,
        output_num_bins=output_num_bins,
        bin_size=bin_size,
        target_index=target_index,
        target_identifier=target_identifier,
        target_description=target_description,
        aggregation=args.aggregation,
        max_genes=args.max_genes,
        min_gene_overlap_fraction=args.min_gene_overlap_fraction,
    )
    out = Path(args.out)
    write_table(rows, out)
    write_rows_csv_gz(rows, out.with_suffix(".csv.gz"))
    summary = manifest_summary(rows)
    summary.update(
        {
            "input_len": input_len,
            "output_num_bins": output_num_bins,
            "bin_size": bin_size,
            "target_index": target_index,
        }
    )
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    out.with_name("target_metadata.json").write_text(
        json.dumps(
            {
                "target_index": target_index,
                "target_identifier": target_identifier,
                "target_description": target_description,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
