from __future__ import annotations

import unittest

import numpy as np

import _bootstrap  # noqa: F401
from pocketreg.borzoi.output_mapping import BorzoiOutputMapper


class OutputMappingTest(unittest.TestCase):
    def test_default_full_borzoi_core_offset(self) -> None:
        mapper = BorzoiOutputMapper(
            input_seq_start=0,
            input_len=524288,
            output_num_bins=6144,
            bin_size=32,
            target_index=0,
        )
        self.assertEqual(mapper.output_core_start, 163840)
        self.assertEqual(mapper.output_core_end, 360448)

    def test_interval_to_bins_and_outside(self) -> None:
        mapper = BorzoiOutputMapper(0, 1000, 10, 10, 0, output_core_start=100)
        overlaps = mapper.genomic_interval_to_bins(105, 125)
        self.assertEqual([x.bin_index for x in overlaps], [0, 1, 2])
        self.assertEqual([x.overlap_bp for x in overlaps], [5, 10, 5])
        self.assertEqual(mapper.genomic_interval_to_bins(0, 50), [])

    def test_aggregate_gene_body(self) -> None:
        mapper = BorzoiOutputMapper(0, 1000, 10, 10, 0, output_core_start=100)
        output = np.ones((10, 1), dtype=float) * 3.0
        result = mapper.aggregate_gene_body(output, 100, 130)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.raw_mean, 3.0)
        self.assertEqual(result.n_bins_used, 3)


if __name__ == "__main__":
    unittest.main()
