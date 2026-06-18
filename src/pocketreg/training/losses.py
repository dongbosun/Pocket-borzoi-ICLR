"""Loss helpers."""

from __future__ import annotations

import numpy as np


def huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    err = np.asarray(y_pred) - np.asarray(y_true)
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    linear = abs_err - quad
    return float(np.mean(0.5 * quad**2 + delta * linear))
