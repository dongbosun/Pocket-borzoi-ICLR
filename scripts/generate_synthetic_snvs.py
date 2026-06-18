#!/usr/bin/env python
"""Generate synthetic SNVs from a gene manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import _path  # noqa: F401

from pocketreg.data.fasta import FastaReader
from pocketreg.data.manifest import read_table, write_table
from pocketreg.data.variants import generate_snvs_for_manifest_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", default="/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/dataset/variants/k562_synthetic_snvs.parquet")
    parser.add_argument("--snvs-per-gene", type=int, default=50)
    parser.add_argument("--region", default="gene_body_plus_tss_flank")
    parser.add_argument("--tss-flank", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-genes", type=int)
    args = parser.parse_args()
    if args.region != "gene_body_plus_tss_flank":
        raise SystemExit("Phase 1 supports --region gene_body_plus_tss_flank only.")
    rng = random.Random(args.seed)
    rows = read_table(args.manifest)
    if args.max_genes:
        rows = rows[: args.max_genes]
    fasta = FastaReader(args.fasta)
    variants = []
    for row in rows:
        variants.extend(
            snv.__dict__
            for snv in generate_snvs_for_manifest_row(
                row,
                fasta=fasta,
                snvs_per_gene=args.snvs_per_gene,
                rng=rng,
                tss_flank=args.tss_flank,
            )
        )
    out = Path(args.out)
    write_table(variants, out)
    summary = {
        "num_variants": len(variants),
        "splits": dict(Counter(v["split"] for v in variants)),
        "chromosomes": dict(Counter(v["chrom"] for v in variants)),
        "ref": dict(Counter(v["ref"] for v in variants)),
        "alt": dict(Counter(v["alt"] for v in variants)),
    }
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
