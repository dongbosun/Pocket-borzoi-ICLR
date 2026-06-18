from __future__ import annotations

import numpy as np

from pocketreg.data.sequence import dna_to_int, gc_content, int_to_onehot, one_hot_encode, reverse_complement


def test_one_hot_acgtn_shape_and_n_zero() -> None:
    onehot = one_hot_encode("ACGTN")
    assert onehot.shape == (4, 5)
    assert np.allclose(onehot[:, 0], [1, 0, 0, 0])
    assert np.allclose(onehot[:, 1], [0, 1, 0, 0])
    assert np.allclose(onehot[:, 2], [0, 0, 1, 0])
    assert np.allclose(onehot[:, 3], [0, 0, 0, 1])
    assert np.allclose(onehot[:, 4], [0, 0, 0, 0])


def test_dna_to_int_and_int_to_onehot() -> None:
    arr = dna_to_int("ACGTNX")
    assert arr.dtype == np.uint8
    assert arr.tolist() == [0, 1, 2, 3, 4, 4]
    assert int_to_onehot(arr).shape == (4, 6)


def test_reverse_complement() -> None:
    assert reverse_complement("ACGTN") == "NACGT"


def test_gc_content() -> None:
    assert gc_content("ACGTNN") == 0.5
    assert np.isnan(gc_content("NNNN"))
