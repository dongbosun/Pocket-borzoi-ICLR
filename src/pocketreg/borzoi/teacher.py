"""Thin wrapper around official Calico Borzoi/Mini-Borzoi K562 models."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pocketreg.borzoi.assets import (
    BorzoiAssetConfig,
    detect_shape_metadata,
    load_params,
    sha256_file,
)
from pocketreg.borzoi.targets import (
    find_k562_rnaseq_candidates,
    parse_targets,
    row_target_index,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BorzoiTeacherMetadata:
    input_len: int
    output_num_bins: int
    output_num_tracks: int
    bin_size: int
    selected_target_index: int
    selected_target_identifier: str | None
    selected_target_description: str | None
    weights_path: str
    params_path: str
    targets_path: str
    weights_sha256: str | None
    params_sha256: str | None
    targets_sha256: str | None
    borzoi_repo: str | None
    baskerville_repo: str | None
    borzoi_git_commit: str | None
    baskerville_git_commit: str | None


def _repo_commit(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _prepend_repo_src(path: Path | None) -> None:
    if path is None:
        return
    candidates = [path / "src", path]
    for candidate in candidates:
        if candidate.exists():
            text = str(candidate.resolve())
            if text not in sys.path:
                sys.path.insert(0, text)


def _select_target(targets_path: Path, target_index: int | None) -> tuple[int, dict[str, Any]]:
    rows = parse_targets(targets_path)
    if target_index is not None:
        for row in rows:
            row_index = row_target_index(row)
            if row_index == target_index:
                return target_index, row
        if 0 <= target_index < len(rows):
            return target_index, rows[target_index]
        raise ValueError(f"Target index {target_index} not present in {targets_path}")

    candidates = find_k562_rnaseq_candidates(rows)
    if not candidates:
        raise ValueError(f"No K562 RNA-seq target candidate found in {targets_path}")
    selected = candidates[0]
    selected_index = row_target_index(selected)
    if selected_index is None:
        selected_index = rows.index(selected)
    if len(candidates) > 1:
        LOGGER.warning(
            "Multiple K562 RNA-seq targets found in %s; using first candidate index %s",
            targets_path,
            selected_index,
        )
    return int(selected_index), selected


class BorzoiK562Teacher:
    """Load official Mini-Borzoi K562 weights and run selected-track inference."""

    def __init__(
        self,
        assets: BorzoiAssetConfig,
        target_index: int | None = None,
        fold: int = 0,
        slice_to_target: bool = True,
    ):
        self.assets = assets
        self.fold = fold
        self.slice_to_target = slice_to_target
        self.weights_path = assets.k562_weights_fold0 if fold == 0 else assets.k562_weights_fold1
        self.params_path = assets.k562_params
        self.targets_path = assets.k562_targets
        for label, path in {
            "weights": self.weights_path,
            "params": self.params_path,
            "targets": self.targets_path,
            "baskerville_repo": assets.baskerville_repo,
        }.items():
            if path is None or not path.exists():
                raise FileNotFoundError(f"Missing Borzoi {label}: {path}")

        _prepend_repo_src(assets.baskerville_repo)
        _prepend_repo_src(assets.borzoi_repo)

        try:
            import tensorflow as tf  # type: ignore
            from baskerville import seqnn  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Could not import TensorFlow/Baskerville for Borzoi teacher. "
                f"PYTHONPATH={sys.path[:5]}, weights={self.weights_path}, "
                f"params={self.params_path}. Original error: {exc}"
            ) from exc

        self.tf = tf
        self.seqnn_module = seqnn
        for gpu in self.tf.config.list_physical_devices("GPU"):
            try:
                self.tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                LOGGER.debug("Could not enable TensorFlow memory growth for %s", gpu)
        self.params = load_params(self.params_path)
        if "model" not in self.params:
            raise ValueError(f"Params file lacks top-level 'model': {self.params_path}")
        self.params_model = dict(self.params["model"])
        self.params_model["verbose"] = False
        self.shape_metadata = detect_shape_metadata(self.params)
        self.selected_target_index, self.selected_target_row = _select_target(
            self.targets_path,
            target_index,
        )

        LOGGER.info("Building Mini-Borzoi model from %s", self.params_path)
        self.seqnn_model = self.seqnn_module.SeqNN(self.params_model)
        self.seqnn_model.restore(str(self.weights_path), head_i=0)
        if slice_to_target:
            self.seqnn_model.build_slice([self.selected_target_index], False)

        output_shape = tuple(int(v) if v is not None else -1 for v in self.seqnn_model.model.output_shape)
        if len(output_shape) != 3:
            raise ValueError(f"Unexpected Borzoi output shape {output_shape}")
        self.input_len = int(self.params_model.get("seq_length") or self.shape_metadata["input_len"])
        self.output_num_bins = int(output_shape[1])
        self.output_num_tracks = int(output_shape[2])
        bin_size = self.shape_metadata.get("bin_size")
        if bin_size is None:
            raise ValueError("Could not infer Borzoi bin_size from params/model metadata")
        self.bin_size = int(bin_size)

    @property
    def mapper_target_index(self) -> int:
        return 0 if self.slice_to_target else self.selected_target_index

    def metadata(self) -> BorzoiTeacherMetadata:
        identifier = (
            self.selected_target_row.get("identifier")
            or self.selected_target_row.get("target")
            or self.selected_target_row.get("index")
        )
        description = self.selected_target_row.get("description") or self.selected_target_row.get(
            "desc"
        )
        return BorzoiTeacherMetadata(
            input_len=self.input_len,
            output_num_bins=self.output_num_bins,
            output_num_tracks=self.output_num_tracks,
            bin_size=self.bin_size,
            selected_target_index=self.selected_target_index,
            selected_target_identifier=str(identifier) if identifier is not None else None,
            selected_target_description=str(description) if description is not None else None,
            weights_path=str(self.weights_path),
            params_path=str(self.params_path),
            targets_path=str(self.targets_path),
            weights_sha256=sha256_file(self.weights_path) if self.weights_path else None,
            params_sha256=sha256_file(self.params_path) if self.params_path else None,
            targets_sha256=sha256_file(self.targets_path) if self.targets_path else None,
            borzoi_repo=str(self.assets.borzoi_repo) if self.assets.borzoi_repo else None,
            baskerville_repo=str(self.assets.baskerville_repo) if self.assets.baskerville_repo else None,
            borzoi_git_commit=_repo_commit(self.assets.borzoi_repo),
            baskerville_git_commit=_repo_commit(self.assets.baskerville_repo),
        )

    def metadata_dict(self) -> dict[str, Any]:
        return asdict(self.metadata())

    def predict(self, one_hot_batch: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        arr = np.asarray(one_hot_batch, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected one-hot batch [B, L, 4], got {arr.shape}")
        if arr.shape[1] != self.input_len or arr.shape[2] != 4:
            raise ValueError(
                f"Expected one-hot shape [B, {self.input_len}, 4], got {arr.shape}"
            )
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        preds = self.seqnn_model.model.predict(arr, verbose=0, **kwargs).astype(np.float32)
        if preds.ndim != 3:
            raise ValueError(f"Expected Borzoi predictions [B, bins, tracks], got {preds.shape}")
        if preds.shape[1] != self.output_num_bins:
            raise ValueError(
                f"Expected {self.output_num_bins} output bins, got {preds.shape[1]}"
            )
        return preds

    def write_metadata_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata_dict(), indent=2, sort_keys=True) + "\n")
