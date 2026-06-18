"""DNA sequence encoding utilities."""

from __future__ import annotations

import numpy as np

DNA_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}
INT_TO_DNA = np.array(["A", "C", "G", "T", "N"], dtype="<U1")
_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def dna_to_int(seq: str) -> np.ndarray:
    """Encode DNA bases as uint8 values A=0, C=1, G=2, T=3, other/N=4."""
    arr = np.fromiter((DNA_TO_INT.get(base.upper(), 4) for base in seq), dtype=np.uint8)
    return arr


def int_to_onehot(arr: np.ndarray) -> np.ndarray:
    """Convert integer-encoded DNA to a float32 one-hot array with shape [4, L]."""
    arr = np.asarray(arr, dtype=np.uint8)
    onehot = np.zeros((4, arr.shape[0]), dtype=np.float32)
    valid = arr < 4
    if np.any(valid):
        onehot[arr[valid], np.nonzero(valid)[0]] = 1.0
    return onehot


def one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode a DNA string into shape [4, L], with N/other all zeros."""
    return int_to_onehot(dna_to_int(seq))


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string."""
    return seq.translate(_RC_TABLE)[::-1].upper()


def gc_content(seq: str) -> float:
    """Return GC fraction over A/C/G/T bases, ignoring N and other bases."""
    seq_upper = seq.upper()
    valid = sum(seq_upper.count(base) for base in "ACGT")
    if valid == 0:
        return float("nan")
    gc = seq_upper.count("G") + seq_upper.count("C")
    return gc / valid
