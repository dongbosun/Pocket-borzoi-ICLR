"""Minimal GTF gene parser for GENCODE-like annotations."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GeneRecord:
    gene_id: str
    gene_name: str
    chrom: str
    start_0based: int
    end_0based: int
    strand: str
    gene_type: str | None = None

    @property
    def tss_0based(self) -> int:
        if self.strand == "+":
            return self.start_0based
        if self.strand == "-":
            return self.end_0based - 1
        return (self.start_0based + self.end_0based) // 2


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def parse_gtf_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in raw.strip().strip(";").split(";"):
        part = part.strip()
        if not part:
            continue
        if " " not in part:
            continue
        key, value = part.split(" ", 1)
        attrs[key] = value.strip().strip('"')
    return attrs


def parse_genes(
    gtf_path: str | Path,
    autosomes_only: bool = True,
    protein_coding_only: bool = True,
) -> list[GeneRecord]:
    path = Path(gtf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    genes: list[GeneRecord] = []
    transcript_gene_bounds: dict[str, dict] = {}
    autosomes = {str(i) for i in range(1, 23)}

    with _open_text(path) as handle:
        for raw in handle:
            if raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs_raw = fields
            chrom_bare = chrom[3:] if chrom.startswith("chr") else chrom
            if autosomes_only and chrom_bare not in autosomes:
                continue
            if feature not in {"gene", "transcript"}:
                continue
            attrs = parse_gtf_attributes(attrs_raw)
            gene_type = (
                attrs.get("gene_type")
                or attrs.get("gene_biotype")
                or attrs.get("transcript_type")
            )
            if protein_coding_only and gene_type and gene_type != "protein_coding":
                continue
            gene_id = attrs.get("gene_id")
            if not gene_id:
                continue
            start_0 = int(start) - 1
            end_0 = int(end)
            if feature == "gene":
                genes.append(
                    GeneRecord(
                        gene_id=gene_id,
                        gene_name=attrs.get("gene_name", gene_id),
                        chrom=chrom,
                        start_0based=start_0,
                        end_0based=end_0,
                        strand=strand,
                        gene_type=gene_type,
                    )
                )
            else:
                current = transcript_gene_bounds.get(gene_id)
                if current is None:
                    transcript_gene_bounds[gene_id] = {
                        "gene_id": gene_id,
                        "gene_name": attrs.get("gene_name", gene_id),
                        "chrom": chrom,
                        "start_0based": start_0,
                        "end_0based": end_0,
                        "strand": strand,
                        "gene_type": gene_type,
                    }
                else:
                    current["start_0based"] = min(current["start_0based"], start_0)
                    current["end_0based"] = max(current["end_0based"], end_0)

    if genes:
        return genes

    # Borzoi helper GTFs such as gencode41_basic_nort_protein.gtf may omit
    # gene rows and contain transcript/exon rows only. Aggregate transcript
    # intervals to gene-level records for manifest construction.
    return [GeneRecord(**record) for record in transcript_gene_bounds.values()]


def iter_exons(gtf_path: str | Path) -> Iterable[tuple[str, int, int, str]]:
    path = Path(gtf_path)
    with _open_text(path) as handle:
        for raw in handle:
            if raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            attrs = parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id")
            if gene_id:
                yield fields[0], int(fields[3]) - 1, int(fields[4]), gene_id
