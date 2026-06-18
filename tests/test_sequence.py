from __future__ import annotations

import unittest

import numpy as np

import _bootstrap  # noqa: F401
from pocketreg.data.sequence import apply_snv, gc_content, one_hot_encode, reverse_complement


class SequenceTest(unittest.TestCase):
    def test_one_hot_n_is_zero(self) -> None:
        arr = one_hot_encode("ACGTN", channels_first=False)
        self.assertEqual(arr.shape, (5, 4))
        np.testing.assert_array_equal(arr[4], np.zeros(4))

    def test_reverse_complement(self) -> None:
        self.assertEqual(reverse_complement("ACGTN"), "NACGT")

    def test_gc_content(self) -> None:
        self.assertAlmostEqual(gc_content("ACGTNN"), 0.5)

    def test_apply_snv(self) -> None:
        result = apply_snv("AACCGG", genomic_pos_0based=102, seq_start_0based=100, ref="C", alt="T")
        self.assertEqual(result.local_index, 2)
        self.assertEqual(result.alt_sequence, "AATCGG")


if __name__ == "__main__":
    unittest.main()
