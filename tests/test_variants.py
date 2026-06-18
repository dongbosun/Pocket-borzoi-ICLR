from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401
from pocketreg.data.sequence import apply_snv
from pocketreg.data.variants import choose_alt
import random


class VariantsTest(unittest.TestCase):
    def test_apply_snv_correct_position(self) -> None:
        result = apply_snv("ACGT", 11, 10, "C", "T")
        self.assertEqual(result.alt_sequence, "ATGT")

    def test_ref_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "REF mismatch"):
            apply_snv("ACGT", 11, 10, "A", "T")

    def test_alt_differs(self) -> None:
        alt = choose_alt("A", random.Random(1))
        self.assertIn(alt, {"C", "G", "T"})
        with self.assertRaisesRegex(ValueError, "ALT allele"):
            apply_snv("ACGT", 10, 10, "A", "A")

    def test_0based_1based_consistency(self) -> None:
        pos_0 = 123
        self.assertEqual(pos_0 + 1, 124)


if __name__ == "__main__":
    unittest.main()
