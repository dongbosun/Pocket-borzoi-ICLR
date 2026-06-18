"""Dataset split helpers."""

from __future__ import annotations

TRAIN_CHROMS = {f"chr{i}" for i in range(1, 17)} | {str(i) for i in range(1, 17)}
VAL_CHROMS = {"chr17", "chr18", "17", "18"}
TEST_CHROMS = {f"chr{i}" for i in range(19, 23)} | {str(i) for i in range(19, 23)}


def chromosome_split(chrom: str) -> str:
    if chrom in TRAIN_CHROMS:
        return "train"
    if chrom in VAL_CHROMS:
        return "val"
    if chrom in TEST_CHROMS:
        return "test"
    return "holdout"


def assert_no_chrom_overlap(rows: list[dict]) -> None:
    split_to_chroms: dict[str, set[str]] = {}
    for row in rows:
        split_to_chroms.setdefault(row["split"], set()).add(row["chrom"])
    seen: dict[str, str] = {}
    for split, chroms in split_to_chroms.items():
        for chrom in chroms:
            if chrom in seen:
                raise AssertionError(f"Chromosome {chrom} appears in both {seen[chrom]} and {split}")
            seen[chrom] = split
