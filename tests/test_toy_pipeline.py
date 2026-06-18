from __future__ import annotations

import json
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401


class ToyPipelineTest(unittest.TestCase):
    def test_smoke_main_outputs_metrics(self) -> None:
        from run_smoke_test import main

        main()
        metrics = Path("/extra/zhanglab1/INDV/dongbos/Pocket-borzoi-ICLR/results/runs/toy_smoke/metrics.json")
        self.assertTrue(metrics.exists())
        data = json.loads(metrics.read_text())
        self.assertIn("track", data)
        self.assertIn("delta", data)


if __name__ == "__main__":
    unittest.main()
