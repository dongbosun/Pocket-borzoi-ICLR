from __future__ import annotations

import importlib.util
import unittest

import _bootstrap  # noqa: F401


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class TrackModelTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        import torch

        from pocketreg.models.small_cnn import SmallCNN, count_parameters

        model = SmallCNN(channels=8, num_blocks=1, stem_stride=2, head_hidden=8)
        x = torch.randn(2, 4, 256)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2,))
        self.assertGreater(count_parameters(model), 0)


if __name__ == "__main__":
    unittest.main()
