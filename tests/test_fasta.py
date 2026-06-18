from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from pocketreg.data.fasta import FastaReader


class FastaTest(unittest.TestCase):
    def test_padding_and_chr_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.fa"
            path.write_text(">1\nACGT\n")
            reader = FastaReader(path)
            self.assertEqual(reader.fetch("chr1", -2, 6), "NNACGTNN")
            self.assertEqual(reader.fetch("1", 1, 3), "CG")


if __name__ == "__main__":
    unittest.main()
