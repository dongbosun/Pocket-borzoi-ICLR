from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from pocketreg.data.gtf import parse_genes, parse_gtf_attributes


class GtfTest(unittest.TestCase):
    def test_parse_attributes_and_genes(self) -> None:
        attrs = parse_gtf_attributes('gene_id "G1"; gene_name "Gene1"; gene_type "protein_coding";')
        self.assertEqual(attrs["gene_id"], "G1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.gtf"
            path.write_text(
                'chr1\tsrc\tgene\t11\t20\t.\t+\t.\tgene_id "G1"; gene_name "Gene1"; gene_type "protein_coding";\n'
                'chrX\tsrc\tgene\t11\t20\t.\t+\t.\tgene_id "GX"; gene_type "protein_coding";\n'
            )
            genes = parse_genes(path)
            self.assertEqual(len(genes), 1)
            self.assertEqual(genes[0].start_0based, 10)
            self.assertEqual(genes[0].end_0based, 20)
            self.assertEqual(genes[0].tss_0based, 10)


if __name__ == "__main__":
    unittest.main()
