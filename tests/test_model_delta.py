from __future__ import annotations

import importlib.util
import unittest

import _bootstrap  # noqa: F401


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class DeltaModelTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        import torch

        from pocketreg.models.delta_siamese_cnn import SiameseDeltaCNN

        model = SiameseDeltaCNN(metadata_dim=3, channels=8, num_blocks=1, stem_stride=2, head_hidden=8)
        ref = torch.randn(2, 4, 256)
        alt = torch.randn(2, 4, 256)
        meta = torch.randn(2, 3)
        y = model(ref, alt, meta)
        self.assertEqual(tuple(y.shape), (2,))
        y.sum().backward()


if __name__ == "__main__":
    unittest.main()
