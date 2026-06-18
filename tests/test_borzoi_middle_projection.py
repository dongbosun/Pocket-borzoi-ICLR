import numpy as np

from pocketreg.borzoi.middle_projection import fit_projection


def test_fit_projection_pls_shapes():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 8)).astype(np.float32)
    y = rng.normal(size=(20, 4)).astype(np.float32)
    train = np.array([True] * 12 + [False] * 8)
    _, model, z, method = fit_projection(x, y, train, n_components=3, method="pls")
    assert method == "pls"
    assert model["method"] == "pls"
    assert z.shape == (20, 3)
