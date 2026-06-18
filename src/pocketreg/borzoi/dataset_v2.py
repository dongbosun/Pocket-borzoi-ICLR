"""Dataset for Pocket-Borzoi v2 track distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pocketreg.data.fasta import FastaReader
from pocketreg.data.manifest import read_table
from pocketreg.data.sequence import one_hot_encode


@dataclass(frozen=True)
class LabelStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_values(cls, values: np.ndarray) -> "LabelStats":
        mean = np.nanmean(values, axis=0).astype(np.float32)
        std = np.nanstd(values, axis=0).astype(np.float32)
        std[std <= 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values.astype(np.float32) - self.mean) / self.std).astype(np.float32)


def _label_cols(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [col for col in frame.columns if col.startswith(prefix)],
        key=lambda name: int(name.split("_")[-1]),
    )


def load_v2_frame(
    manifest_path: str | Path,
    rich_labels_path: str | Path,
    profile_pca_path: str | Path,
    aux_pca_path: str | Path,
    middle_projection_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    manifest = pd.DataFrame(read_table(manifest_path))
    rich = pd.read_parquet(rich_labels_path)
    profile = pd.read_parquet(profile_pca_path)
    aux = pd.read_parquet(aux_pca_path)
    middle = pd.read_parquet(middle_projection_path)
    base_cols = [
        "example_id",
        "gene_id",
        "gene_name",
        "chrom",
        "gene_start",
        "gene_end",
        "strand",
        "tss",
        "split",
    ]
    frame = manifest[base_cols].merge(
        rich[
            [
                "example_id",
                "primary_0_q_mean",
                "primary_0_q_fold0",
                "primary_0_q_fold1",
                "status",
            ]
        ],
        on="example_id",
        how="inner",
    )
    frame = frame.merge(profile.drop(columns=[c for c in ("gene_id", "split") if c in profile.columns]), on="example_id", how="inner")
    frame = frame.merge(aux.drop(columns=[c for c in ("gene_id", "split") if c in aux.columns]), on="example_id", how="inner")
    frame = frame.merge(middle.drop(columns=[c for c in ("gene_id", "split") if c in middle.columns]), on="example_id", how="inner")
    cols = {
        "fold": ["primary_0_q_fold0", "primary_0_q_fold1"],
        "profile_pca": _label_cols(frame, "profile_pca_"),
        "aux_pca": _label_cols(frame, "aux_pca_"),
        "middle_proj": _label_cols(frame, "middle_proj_"),
    }
    return frame, cols


def compute_label_stats(frame: pd.DataFrame, cols: dict[str, list[str]], train_mask: np.ndarray) -> dict[str, LabelStats]:
    stats = {
        "primary": LabelStats.from_values(frame.loc[train_mask, ["primary_0_q_mean"]].to_numpy(dtype=np.float32)),
        "fold": LabelStats.from_values(frame.loc[train_mask, cols["fold"]].to_numpy(dtype=np.float32)),
        "profile_pca": LabelStats.from_values(frame.loc[train_mask, cols["profile_pca"]].to_numpy(dtype=np.float32)),
        "aux_pca": LabelStats.from_values(frame.loc[train_mask, cols["aux_pca"]].to_numpy(dtype=np.float32)),
        "middle_proj": LabelStats.from_values(frame.loc[train_mask, cols["middle_proj"]].to_numpy(dtype=np.float32)),
    }
    return stats


class BorzoiV2Dataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        label_cols: dict[str, list[str]],
        stats: dict[str, LabelStats],
        fasta_path: str | Path,
        context_len: int,
        standardize: bool = True,
        max_rows: int | None = None,
    ):
        if max_rows is not None:
            frame = frame.head(max_rows).copy()
        self.frame = frame.reset_index(drop=True)
        self.rows = self.frame.to_dict(orient="records")
        self.label_cols = label_cols
        self.stats = stats
        self.fasta_path = Path(fasta_path)
        self._fasta: FastaReader | None = None
        self.context_len = int(context_len)
        self.standardize = standardize

    @property
    def fasta(self) -> FastaReader:
        if self._fasta is None:
            self._fasta = FastaReader(self.fasta_path)
        return self._fasta

    def __len__(self) -> int:
        return len(self.rows)

    def _mask(self, row: dict[str, Any], start: int, end: int) -> np.ndarray:
        mask = np.zeros((self.context_len,), dtype=np.float32)
        gene_start = max(start, int(row["gene_start"]))
        gene_end = min(end, int(row["gene_end"]))
        if gene_end > gene_start:
            mask[gene_start - start : gene_end - start] = 1.0
        tss = int(row["tss"])
        if start <= tss < end:
            local = tss - start
            left = max(0, local - 128)
            right = min(self.context_len, local + 129)
            mask[left:right] = 1.0
        return mask

    def _labels(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        values = {
            "primary": np.asarray([row["primary_0_q_mean"]], dtype=np.float32),
            "fold": np.asarray([row[col] for col in self.label_cols["fold"]], dtype=np.float32),
            "profile_pca": np.asarray([row[col] for col in self.label_cols["profile_pca"]], dtype=np.float32),
            "aux_pca": np.asarray([row[col] for col in self.label_cols["aux_pca"]], dtype=np.float32),
            "middle_proj": np.asarray([row[col] for col in self.label_cols["middle_proj"]], dtype=np.float32),
        }
        if self.standardize:
            values = {key: self.stats[key].transform(value) for key, value in values.items()}
        return values

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        tss = int(row["tss"])
        start = tss - self.context_len // 2
        end = start + self.context_len
        seq = self.fasta.fetch(str(row["chrom"]), start, end, pad=True)
        x4 = one_hot_encode(seq, channels_first=True)
        mask = self._mask(row, start, end)[None, :]
        x = np.concatenate([x4, mask], axis=0)
        labels = self._labels(row)
        return {
            "x": torch.from_numpy(x),
            "primary": torch.tensor(float(labels["primary"][0]), dtype=torch.float32),
            "fold": torch.from_numpy(labels["fold"]),
            "profile_pca": torch.from_numpy(labels["profile_pca"]),
            "aux_pca": torch.from_numpy(labels["aux_pca"]),
            "middle_proj": torch.from_numpy(labels["middle_proj"]),
            "example_id": str(row["example_id"]),
            "primary_raw": float(row["primary_0_q_mean"]),
        }
