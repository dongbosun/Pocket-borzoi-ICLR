"""DNA sequence utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DNA_ALPHABET = ("A", "C", "G", "T")
BASE_TO_INDEX = {base: i for i, base in enumerate(DNA_ALPHABET)}
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def normalize_dna(seq: str) -> str:
    return seq.upper()


def one_hot_encode(seq: str, channels_first: bool = True) -> np.ndarray:
    """One-hot encode A/C/G/T. N and other bases are all zeros."""

    seq = normalize_dna(seq)
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = BASE_TO_INDEX.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr.T if channels_first else arr


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1].upper()


def gc_content(seq: str) -> float:
    seq = normalize_dna(seq)
    acgt = sum(seq.count(base) for base in DNA_ALPHABET)
    if acgt == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / acgt


@dataclass(frozen=True)
class VariantApplyResult:
    alt_sequence: str
    local_index: int
    ref_observed: str


def apply_snv(
    seq: str,
    genomic_pos_0based: int,
    seq_start_0based: int,
    ref: str,
    alt: str,
    skip_ref_check: bool = False,
) -> VariantApplyResult:
    """Apply one SNV to a sequence window using 0-based genomic coordinates."""

    ref = ref.upper()
    alt = alt.upper()
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError("Only single-nucleotide SNVs are supported.")
    if ref not in BASE_TO_INDEX or alt not in BASE_TO_INDEX:
        raise ValueError("REF and ALT must be A/C/G/T.")
    if alt == ref:
        raise ValueError("ALT allele must differ from REF allele.")

    local = genomic_pos_0based - seq_start_0based
    if local < 0 or local >= len(seq):
        raise ValueError(
            f"Variant position {genomic_pos_0based} is outside sequence window "
            f"[{seq_start_0based}, {seq_start_0based + len(seq)})."
        )
    observed = seq[local].upper()
    if not skip_ref_check and observed != ref:
        raise ValueError(
            f"REF mismatch at local index {local}: expected {ref}, observed {observed}."
        )
    alt_seq = seq[:local] + alt + seq[local + 1 :]
    return VariantApplyResult(alt_sequence=alt_seq, local_index=local, ref_observed=observed)
