"""Gene manifest construction helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pocketreg.borzoi.output_mapping import BorzoiOutputMapper
from pocketreg.data.fasta import FastaReader
from pocketreg.data.gtf import GeneRecord, parse_genes
from pocketreg.data.splits import chromosome_split


def build_gene_manifest_rows(
    genes: list[GeneRecord],
    fasta: FastaReader,
    input_len: int,
    output_num_bins: int,
    bin_size: int,
    target_index: int,
    target_identifier: str = "",
    target_description: str = "",
    aggregation: str = "gene_body_log1p_mean",
    max_genes: int | None = None,
    min_gene_overlap_fraction: float = 0.8,
    source: str = "gencode",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gene in genes:
        try:
            fasta.normalize_chrom(gene.chrom)
        except KeyError:
            continue
        tss = gene.tss_0based
        seq_start = tss - input_len // 2
        seq_end = seq_start + input_len
        mapper = BorzoiOutputMapper(
            input_seq_start=seq_start,
            input_len=input_len,
            output_num_bins=output_num_bins,
            bin_size=bin_size,
            target_index=target_index,
        )
        overlaps = mapper.genomic_interval_to_bins(gene.start_0based, gene.end_0based)
        overlap_bp = sum(item.overlap_bp for item in overlaps)
        gene_len = max(1, gene.end_0based - gene.start_0based)
        overlap_fraction = overlap_bp / gene_len
        if overlap_fraction < min_gene_overlap_fraction:
            continue
        split = chromosome_split(gene.chrom)
        if split == "holdout":
            continue
        rows.append(
            {
                "example_id": f"{gene.gene_id}|{gene.chrom}:{seq_start}-{seq_end}",
                "gene_id": gene.gene_id,
                "gene_name": gene.gene_name,
                "chrom": gene.chrom,
                "gene_start": gene.start_0based,
                "gene_end": gene.end_0based,
                "strand": gene.strand,
                "tss": tss,
                "seq_start": seq_start,
                "seq_end": seq_end,
                "input_len": input_len,
                "output_core_start": mapper.output_core_start,
                "output_core_end": mapper.output_core_end,
                "output_bin_size": bin_size,
                "output_num_bins": output_num_bins,
                "target_index": target_index,
                "target_identifier": target_identifier,
                "target_description": target_description,
                "label_source": "borzoi_teacher_pseudolabel",
                "split": split,
                "source": source,
                "aggregation_mode": aggregation,
                "gene_overlap_fraction": overlap_fraction,
                "n_bins_overlapping_gene": len(overlaps),
            }
        )
    if max_genes and len(rows) > max_genes:
        rows = select_balanced_rows(rows, max_genes)
    return rows


def select_balanced_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    """Select up to max_rows while preserving train/val/test coverage."""

    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    split_order = ["train", "val", "test"]
    by_split = {split: [row for row in rows if row.get("split") == split] for split in split_order}
    selected: list[dict[str, Any]] = []
    base_quota = max(1, max_rows // len(split_order))
    used_ids: set[int] = set()
    for split in split_order:
        for row in by_split[split][:base_quota]:
            selected.append(row)
            used_ids.add(id(row))
            if len(selected) >= max_rows:
                return selected
    # Fill remaining slots in original order after each split has representation.
    for row in rows:
        if id(row) in used_ids:
            continue
        selected.append(row)
        if len(selected) >= max_rows:
            break
    return selected


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "chromosomes": dict(Counter(row["chrom"] for row in rows)),
    }


def parse_genes_from_gtf(
    gtf: str | Path,
    autosomes_only: bool = True,
    protein_coding_only: bool = True,
) -> list[GeneRecord]:
    return parse_genes(gtf, autosomes_only=autosomes_only, protein_coding_only=protein_coding_only)
