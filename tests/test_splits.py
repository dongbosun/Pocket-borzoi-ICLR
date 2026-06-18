from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from pocketreg.data.splits import assert_no_chrom_overlap, chromosome_split


class SplitsTest(unittest.TestCase):
    def test_chromosome_split(self) -> None:
        self.assertEqual(chromosome_split("chr1"), "train")
        self.assertEqual(chromosome_split("chr17"), "val")
        self.assertEqual(chromosome_split("chr22"), "test")

    def test_no_overlap(self) -> None:
        assert_no_chrom_overlap([{"split": "train", "chrom": "chr1"}, {"split": "test", "chrom": "chr22"}])

    def test_overlap_raises(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_chrom_overlap([{"split": "train", "chrom": "chr1"}, {"split": "val", "chrom": "chr1"}])


if __name__ == "__main__":
    unittest.main()
