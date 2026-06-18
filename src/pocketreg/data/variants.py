"""Variant helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .fasta import FastaReader
from .sequence import DNA_ALPHABET


@dataclass(frozen=True)
class SyntheticSnv:
    variant_example_id: str
    example_id: str
    gene_id: str
    chrom: str
    pos_0based: int
    pos_1based: int
    ref: str
    alt: str
    distance_to_tss: int
    split: str
    synthetic_or_real: str = "synthetic"
    variant_type: str = "SNV"


def choose_alt(ref: str, rng: random.Random) -> str:
    ref = ref.upper()
    choices = [base for base in DNA_ALPHABET if base != ref]
    if len(choices) != 3:
        raise ValueError(f"Invalid REF allele for SNV: {ref}")
    return rng.choice(choices)


def generate_snvs_for_manifest_row(
    row: dict,
    fasta: FastaReader,
    snvs_per_gene: int,
    rng: random.Random,
    tss_flank: int = 32768,
) -> list[SyntheticSnv]:
    chrom = row["chrom"]
    seq_start = int(row["seq_start"])
    seq_end = int(row["seq_end"])
    tss = int(row["tss"])
    region_start = max(seq_start, int(row["gene_start"]) - tss_flank, tss - tss_flank)
    region_end = min(seq_end, int(row["gene_end"]) + tss_flank, tss + tss_flank + 1)
    if region_end <= region_start:
        return []
    seq = fasta.fetch(chrom, region_start, region_end, pad=False)
    candidate_positions = [
        region_start + i for i, base in enumerate(seq.upper()) if base in DNA_ALPHABET
    ]
    if not candidate_positions:
        return []
    if len(candidate_positions) <= snvs_per_gene:
        sampled = candidate_positions
    else:
        sampled = rng.sample(candidate_positions, snvs_per_gene)
    snvs: list[SyntheticSnv] = []
    for j, pos in enumerate(sorted(sampled)):
        ref = fasta.fetch(chrom, pos, pos + 1, pad=False)
        alt = choose_alt(ref, rng)
        snvs.append(
            SyntheticSnv(
                variant_example_id=f"{row['example_id']}::snv{j:04d}",
                example_id=row["example_id"],
                gene_id=row["gene_id"],
                chrom=chrom,
                pos_0based=pos,
                pos_1based=pos + 1,
                ref=ref,
                alt=alt,
                distance_to_tss=pos - tss,
                split=row.get("split", "unknown"),
            )
        )
    return snvs
