"""FASTA reader with chromosome normalization and edge padding."""

from __future__ import annotations

from pathlib import Path


class FastaReader:
    """Small FASTA reader.

    Uses pyfaidx when available so real hg38 stays on disk. Falls back to a
    simple in-memory parser for toy tests and minimal environments.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._faidx = None
        try:
            from pyfaidx import Fasta  # type: ignore

            self._faidx = Fasta(str(self.path), as_raw=True, sequence_always_upper=True)
            self._records = None
        except Exception:
            self._records = self._load_records()
        self._name_map = self._build_name_map()

    def _load_records(self) -> dict[str, str]:
        records: dict[str, list[str]] = {}
        current: str | None = None
        with self.path.open() as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    current = line[1:].split()[0]
                    records[current] = []
                elif current is None:
                    raise ValueError(f"Malformed FASTA {self.path}: sequence before header")
                else:
                    records[current].append(line.upper())
        return {name: "".join(parts) for name, parts in records.items()}

    def _build_name_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        names = self._faidx.keys() if self._faidx is not None else self._records
        for name in names:
            mapping[name] = name
            if name.startswith("chr"):
                mapping[name[3:]] = name
            else:
                mapping[f"chr{name}"] = name
        return mapping

    @property
    def chroms(self) -> set[str]:
        if self._faidx is not None:
            return set(self._faidx.keys())
        return set(self._records)

    def normalize_chrom(self, chrom: str) -> str:
        if chrom in self._name_map:
            return self._name_map[chrom]
        raise KeyError(f"Chromosome {chrom!r} not found in FASTA {self.path}")

    def chrom_length(self, chrom: str) -> int:
        chrom = self.normalize_chrom(chrom)
        if self._faidx is not None:
            return len(self._faidx[chrom])
        return len(self._records[chrom])

    def fetch(self, chrom: str, start: int, end: int, pad: bool = True) -> str:
        """Fetch [start, end) using 0-based coordinates and optional N padding."""

        if end < start:
            raise ValueError(f"Invalid interval [{start}, {end})")
        requested_len = end - start
        chrom = self.normalize_chrom(chrom)
        chrom_len = self.chrom_length(chrom)
        clipped_start = max(start, 0)
        clipped_end = min(end, chrom_len)

        if not pad and (start < 0 or end > chrom_len):
            raise ValueError(f"Interval [{start}, {end}) is outside {chrom}:{chrom_len}")

        left_pad = "N" * max(0, -start)
        right_pad = "N" * max(0, end - chrom_len)
        if clipped_end > clipped_start:
            if self._faidx is not None:
                body = self._faidx[chrom][clipped_start:clipped_end]
            else:
                body = self._records[chrom][clipped_start:clipped_end]
        else:
            body = ""
        result = left_pad + body + right_pad
        if len(result) != requested_len:
            raise AssertionError(
                f"FASTA fetch length mismatch: expected {requested_len}, got {len(result)}"
            )
        return result.upper()
