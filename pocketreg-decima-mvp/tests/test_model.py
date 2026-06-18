from __future__ import annotations

import torch

from pocketreg.models.small_cnn import build_model, count_parameters


def test_tiny_model_forward_and_backward() -> None:
    model = build_model(
        {
            "preset": "tiny_100k",
            "channels": 16,
            "num_blocks": 2,
            "head_hidden": 32,
            "norm": "groupnorm",
        }
    )
    x = torch.randn(2, 4, 1024)
    y = model(x)
    assert y.shape == (2,)
    loss = y.square().mean()
    loss.backward()
    assert count_parameters(model) > 0
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
