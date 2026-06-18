"""Split assignment utilities."""

from __future__ import annotations

import pandas as pd

TRAIN_CHROMS = {str(i) for i in range(1, 17)}
VAL_CHROMS = {"17", "18"}
TEST_CHROMS = {"19", "20", "21", "22"}
SEX_CHROMS = {"X", "Y"}


def chrom_key(chrom: str) -> str:
    """Return a normalized chromosome key without a chr prefix."""
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value.upper()


def chromosome_split_for_chrom(chrom: str, include_sex_chromosomes: bool = False) -> str | None:
    """Assign one chromosome to train/val/test under the default held-out split."""
    key = chrom_key(chrom)
    if key in TRAIN_CHROMS:
        return "train"
    if key in VAL_CHROMS:
        return "val"
    if key in TEST_CHROMS:
        return "test"
    if include_sex_chromosomes and key in SEX_CHROMS:
        return "train"
    return None


def assign_chromosome_splits(
    chroms: pd.Series, include_sex_chromosomes: bool = False
) -> pd.Series:
    """Assign chromosome-based splits for a chromosome series."""
    return chroms.map(lambda c: chromosome_split_for_chrom(c, include_sex_chromosomes))


def assert_no_chromosome_overlap(manifest: pd.DataFrame) -> None:
    """Raise if any chromosome key appears in multiple splits."""
    if "chrom" not in manifest or "split" not in manifest:
        raise ValueError("Manifest must contain chrom and split columns.")
    grouped = manifest.assign(_chrom_key=manifest["chrom"].map(chrom_key)).groupby("_chrom_key")[
        "split"
    ]
    overlaps = {chrom: sorted(set(vals.dropna())) for chrom, vals in grouped}
    overlaps = {chrom: vals for chrom, vals in overlaps.items() if len(vals) > 1}
    if overlaps:
        raise ValueError(f"Chromosome split leakage detected: {overlaps}")


def split_counts_by_chromosome(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return counts by split and chromosome."""
    return (
        manifest.assign(chrom_key=manifest["chrom"].map(chrom_key))
        .groupby(["split", "chrom_key"], dropna=False)
        .size()
        .reset_index(name="n")
    )
