"""Dataset and target shaping for Borzoi delta v2 training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pocketreg.data.fasta import FastaReader
from pocketreg.data.manifest import read_table
from pocketreg.data.sequence import one_hot_encode


BASES = ("A", "C", "G", "T")


def asinh_transform(delta: np.ndarray | float, scale: float) -> np.ndarray | float:
    scale = float(scale) if float(scale) > 0 else 1e-3
    return np.arcsinh(np.asarray(delta) / scale)


def inverse_asinh_transform(value: np.ndarray | float, scale: float) -> np.ndarray | float:
    scale = float(scale) if float(scale) > 0 else 1e-3
    return np.sinh(np.asarray(value)) * scale


def build_delta_metadata(row: dict[str, Any]) -> np.ndarray:
    dist = float(row.get("distance_to_tss", 0.0))
    gene_start = float(row.get("gene_start", row.get("pos_0based", 0.0)))
    gene_end = float(row.get("gene_end", gene_start + 1.0))
    gene_len = max(1.0, gene_end - gene_start)
    pos = float(row.get("pos_0based", gene_start))
    rel = (pos - gene_start) / gene_len
    strand = 1.0 if row.get("strand") == "+" else -1.0
    ref = str(row.get("ref", "N")).upper()
    alt = str(row.get("alt", "N")).upper()
    ref_one = [1.0 if ref == b else 0.0 for b in BASES]
    alt_one = [1.0 if alt == b else 0.0 for b in BASES]
    return np.array(
        [
            dist / 100000.0,
            np.sign(dist) * np.log1p(abs(dist)) / 12.0,
            rel,
            np.log1p(gene_len) / 12.0,
            strand,
            *ref_one,
            *alt_one,
        ],
        dtype=np.float32,
    )


def load_delta_v2_frame(delta_labels: Path, manifest: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(delta_labels)
    if "status" not in df:
        df["status"] = "success"
    df = df[df["status"] == "success"].copy()
    if manifest is not None:
        manifest_df = pd.DataFrame(read_table(manifest))
        keep = [
            "example_id",
            "gene_name",
            "gene_start",
            "gene_end",
            "strand",
            "tss",
            "seq_start",
            "seq_end",
            "input_len",
            "output_core_start",
            "output_core_end",
        ]
        keep = [c for c in keep if c in manifest_df.columns]
        df = df.merge(manifest_df[keep], on="example_id", how="left")
    return df


def estimate_delta_scale(train_delta: np.ndarray, quantile: float = 0.90, default: float = 1e-3) -> float:
    values = np.abs(np.asarray(train_delta, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    scale = float(np.quantile(values, quantile))
    if not np.isfinite(scale) or scale <= 0:
        return default
    return max(scale, default / 100.0)


def compute_sample_weights(abs_delta: np.ndarray, scale: float, alpha: float = 4.0, cap: float = 20.0) -> np.ndarray:
    scaled = np.asarray(abs_delta, dtype=np.float64) / max(float(scale), 1e-12)
    weights = 1.0 + float(alpha) * np.minimum(scaled, float(cap))
    return weights.astype(np.float32)


@dataclass(frozen=True)
class DeltaV2Stats:
    target_scale: float
    effect_threshold: float
    metadata_dim: int


class DeltaV2Dataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        fasta_path: Path,
        context_len: int,
        target_scale: float,
        effect_threshold: float,
        metadata_features: bool = True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.rows = self.frame.to_dict(orient="records")
        self.fasta_path = Path(fasta_path)
        self.context_len = int(context_len)
        self.target_scale = float(target_scale)
        self.effect_threshold = float(effect_threshold)
        self.metadata_features = bool(metadata_features)
        self._fasta: FastaReader | None = None

    @property
    def fasta(self) -> FastaReader:
        if self._fasta is None:
            self._fasta = FastaReader(self.fasta_path)
        return self._fasta

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        pos = int(row["pos_0based"])
        start = pos - self.context_len // 2
        end = start + self.context_len
        ref_seq = self.fasta.fetch(str(row["chrom"]), start, end, pad=True)
        center = pos - start
        observed = ref_seq[center].upper()
        expected = str(row["ref"]).upper()
        if observed != expected:
            raise ValueError(
                f"REF mismatch for {row.get('variant_example_id')}: expected {expected}, observed {observed}"
            )
        alt = str(row["alt"]).upper()
        if alt == expected:
            raise ValueError(f"ALT equals REF for {row.get('variant_example_id')}: {alt}")
        alt_seq = ref_seq[:center] + alt + ref_seq[center + 1 :]
        delta = float(row["delta_teacher"])
        y = float(asinh_transform(delta, self.target_scale))
        metadata = build_delta_metadata(row) if self.metadata_features else np.zeros((0,), dtype=np.float32)
        effect = 1.0 if abs(delta) >= self.effect_threshold else 0.0
        weight = float(row.get("sample_weight", 1.0))
        return {
            "seq_ref": torch.from_numpy(one_hot_encode(ref_seq, channels_first=True)),
            "seq_alt": torch.from_numpy(one_hot_encode(alt_seq, channels_first=True)),
            "metadata": torch.from_numpy(metadata),
            "target": torch.tensor(y, dtype=torch.float32),
            "effect": torch.tensor(effect, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "delta_raw": torch.tensor(delta, dtype=torch.float32),
            "variant_example_id": str(row["variant_example_id"]),
            "example_id": str(row.get("example_id", "")),
        }
