from __future__ import annotations

import pandas as pd

from pocketreg.data.splits import assert_no_chromosome_overlap, assign_chromosome_splits


def test_chromosome_split_no_overlap() -> None:
    chroms = pd.Series([f"chr{i}" for i in range(1, 23)])
    manifest = pd.DataFrame({"chrom": chroms, "split": assign_chromosome_splits(chroms)})
    assert_no_chromosome_overlap(manifest)
    assert set(manifest["split"].dropna()) == {"train", "val", "test"}


def test_split_counts_nonzero_for_toy_chroms() -> None:
    chroms = pd.Series([f"chr{i}" for i in range(1, 23)] * 2)
    splits = assign_chromosome_splits(chroms)
    counts = splits.value_counts().to_dict()
    assert counts["train"] > 0
    assert counts["val"] > 0
    assert counts["test"] > 0
