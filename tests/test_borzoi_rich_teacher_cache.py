import numpy as np

from pocketreg.borzoi.rich_teacher_cache import downsample_profile


def test_downsample_profile_average_pools_to_requested_bins():
    x = np.arange(16, dtype=np.float32)
    y = downsample_profile(x, 4)
    np.testing.assert_allclose(y.astype(np.float32), np.array([1.5, 5.5, 9.5, 13.5], dtype=np.float32))
    assert y.dtype == np.float16
