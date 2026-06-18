from __future__ import annotations

from pathlib import Path

from pocketreg.data.fasta import FastaReader


def write_test_fasta(path: Path) -> None:
    path.write_text(">1\nACGTACGT\n>chr2\nTTTTCCCC\n")


def test_fetch_returns_requested_length(tmp_path: Path) -> None:
    fasta = tmp_path / "toy.fa"
    write_test_fasta(fasta)
    reader = FastaReader(fasta)
    assert reader.fetch("1", 0, 4) == "ACGT"
    assert len(reader.fetch("1", 0, 20)) == 20


def test_negative_start_and_end_padding(tmp_path: Path) -> None:
    fasta = tmp_path / "toy.fa"
    write_test_fasta(fasta)
    reader = FastaReader(fasta)
    assert reader.fetch("1", -2, 3) == "NNACG"
    assert reader.fetch("1", 6, 11) == "GTNNN"


def test_chr_prefix_normalization(tmp_path: Path) -> None:
    fasta = tmp_path / "toy.fa"
    write_test_fasta(fasta)
    reader = FastaReader(fasta)
    assert reader.normalize_chrom("chr1") == "1"
    assert reader.normalize_chrom("2") == "chr2"
