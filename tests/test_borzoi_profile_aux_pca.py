import numpy as np

from pocketreg.borzoi.profile_pca import fit_standardized_pca


def test_fit_standardized_pca_uses_train_mask_and_shapes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 6)).astype(np.float32)
    train_mask = np.array([True, True, True, True, True, False, False, False, False, False])
    scaler, pca, z = fit_standardized_pca(x, train_mask, n_components=3, random_state=7)
    assert z.shape == (10, 3)
    assert pca.n_components_ == 3
    train_z = scaler.transform(x[train_mask])
    np.testing.assert_allclose(train_z.mean(axis=0), np.zeros(6), atol=1e-6)
