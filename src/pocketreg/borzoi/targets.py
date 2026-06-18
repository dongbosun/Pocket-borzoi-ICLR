"""Target metadata parsing."""

from __future__ import annotations

import csv
from pathlib import Path


def parse_targets(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(handle, dialect=dialect)
        rows = []
        for i, row in enumerate(reader):
            clean = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if "index" not in clean:
                clean["index"] = str(i)
            rows.append(clean)
    return rows


def find_k562_rnaseq_candidates(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        haystack = " ".join(str(value) for value in row.values()).lower()
        if "k562" in haystack and ("rna" in haystack or "rna-seq" in haystack):
            candidates.append(row)
    return candidates


def row_target_index(row: dict) -> int | None:
    for key in ("index", "target", "target_index", "identifier"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
