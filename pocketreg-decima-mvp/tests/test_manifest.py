from __future__ import annotations

from pathlib import Path

import pytest

from pocketreg.data.anndata_utils import get_teacher_labels, select_pseudobulk
from pocketreg.data.manifest import build_manifest_from_anndata, save_manifest_outputs


def test_teacher_label_extraction_shape(tiny_anndata) -> None:
    labels = get_teacher_labels(tiny_anndata, 0, "preds")
    assert labels.shape == (tiny_anndata.n_vars,)


def test_invalid_layer_raises_clear_error(tiny_anndata) -> None:
    with pytest.raises(KeyError, match="Available layers"):
        get_teacher_labels(tiny_anndata, 0, "missing")


def test_toy_anndata_manifest_builds_and_saves_metadata(tiny_anndata, tmp_path: Path) -> None:
    idx, metadata = select_pseudobulk(tiny_anndata, target_index=0)
    manifest, target_metadata, summary = build_manifest_from_anndata(
        tiny_anndata,
        idx,
        "preds",
        128,
        split_mode="chromosome",
        skip_fasta_check=True,
    )
    assert len(manifest) == tiny_anndata.n_vars
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert target_metadata["target_obs_idx"] == metadata["target_obs_idx"]
    out = tmp_path / "manifest.parquet"
    save_manifest_outputs(manifest, out, target_metadata, summary)
    assert out.exists()
    assert (tmp_path / "target_metadata.json").exists()
    assert (tmp_path / "manifest_summary.json").exists()
