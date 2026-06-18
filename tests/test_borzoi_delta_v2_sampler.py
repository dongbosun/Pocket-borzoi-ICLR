from __future__ import annotations

import numpy as np

from pocketreg.borzoi.delta_dataset_v2 import (
    asinh_transform,
    compute_sample_weights,
    inverse_asinh_transform,
)
from pocketreg.models.delta_v2 import DeltaSiameseV2


def test_asinh_roundtrip():
    values = np.array([-0.01, -1e-4, 0.0, 2e-4, 0.02])
    z = asinh_transform(values, scale=1e-3)
    recovered = inverse_asinh_transform(z, scale=1e-3)
    assert np.allclose(values, recovered)


def test_sample_weights_prioritize_large_effects():
    abs_delta = np.array([0.0, 1e-5, 1e-3, 1e-2])
    weights = compute_sample_weights(abs_delta, scale=1e-3, alpha=2.0, cap=5.0)
    assert weights[0] == 1.0
    assert weights[-1] > weights[1]
    assert weights[-1] <= 11.0


def test_delta_v2_model_forward_backward():
    import torch

    model = DeltaSiameseV2(metadata_dim=13, channels=16, num_blocks=2, stem_stride=4, head_hidden=32)
    ref = torch.randn(3, 4, 1024)
    alt = torch.randn(3, 4, 1024)
    meta = torch.randn(3, 13)
    out = model(ref, alt, meta)
    assert out["delta"].shape == (3,)
    assert out["effect_logit"].shape == (3,)
    loss = out["delta"].square().mean() + out["effect_logit"].square().mean()
    loss.backward()
