import numpy as np

from pocketreg.borzoi.middle_features import pooled_spatial_features


def test_pooled_spatial_features_mean_max_center():
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    y = pooled_spatial_features(x, center_bins=2).astype(np.float32)
    assert y.shape == (12,)
    np.testing.assert_allclose(y[:4], x.mean(axis=0))
    np.testing.assert_allclose(y[4:8], x.max(axis=0))
    np.testing.assert_allclose(y[8:], x[2:4].mean(axis=0))
