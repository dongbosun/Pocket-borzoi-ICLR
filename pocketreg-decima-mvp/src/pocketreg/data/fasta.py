"""FASTA access with chromosome-name normalization and boundary padding."""

from __future__ import annotations

from pathlib import Path

from pyfaidx import Fasta

_VALID_BASES = set("ACGT")


class FastaReader:
    """Small wrapper around pyfaidx for fixed-length sequence windows."""

    def __init__(self, fasta_path: str | Path):
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
        self.fasta = Fasta(str(self.fasta_path), as_raw=True, sequence_always_upper=True)
        self.chroms = set(self.fasta.keys())

    def normalize_chrom(self, chrom: str) -> str:
        """Map chr-prefixed and unprefixed chromosome names to the FASTA names."""
        chrom_str = str(chrom)
        if chrom_str in self.chroms:
            return chrom_str
        if chrom_str.startswith("chr") and chrom_str[3:] in self.chroms:
            return chrom_str[3:]
        prefixed = f"chr{chrom_str}"
        if prefixed in self.chroms:
            return prefixed
        if chrom_str in {"M", "MT", "chrM", "chrMT"}:
            for candidate in ("chrM", "MT", "M"):
                if candidate in self.chroms:
                    return candidate
        examples = ", ".join(sorted(list(self.chroms))[:8])
        raise KeyError(
            f"Chromosome {chrom!r} was not found in {self.fasta_path}. "
            f"Example FASTA chromosomes: {examples}"
        )

    def fetch(self, chrom: str, start: int, end: int, pad: bool = True) -> str:
        """Fetch [start, end) sequence, padding out-of-bound regions with N."""
        if end < start:
            raise ValueError(f"Invalid interval: start={start}, end={end}")
        requested_len = int(end) - int(start)
        fasta_chrom = self.normalize_chrom(chrom)
        chrom_len = len(self.fasta[fasta_chrom])

        if not pad and (start < 0 or end > chrom_len):
            raise ValueError(
                f"Interval {chrom}:{start}-{end} exceeds chromosome bounds "
                f"0-{chrom_len}; pass pad=True to allow N padding."
            )

        fetch_start = max(0, int(start))
        fetch_end = min(int(end), chrom_len)
        left_pad = max(0, -int(start))
        right_pad = max(0, int(end) - chrom_len)

        if fetch_end > fetch_start:
            body = str(self.fasta[fasta_chrom][fetch_start:fetch_end])
        else:
            body = ""
        seq = ("N" * left_pad) + body + ("N" * right_pad)
        if pad and len(seq) != requested_len:
            seq = seq[:requested_len].ljust(requested_len, "N")
        return "".join(base if base in _VALID_BASES else "N" for base in seq.upper())
