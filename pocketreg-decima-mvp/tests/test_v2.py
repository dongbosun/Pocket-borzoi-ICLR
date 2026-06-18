from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pocketreg.data.v2 import DecimaV2GeneSequenceDataset, build_v2_manifest_from_anndata
from pocketreg.models.tcn import build_v2_model


def test_v2_manifest_columns(tiny_anndata):
    manifest, _, summary = build_v2_manifest_from_anndata(
        tiny_anndata,
        primary_idx=0,
        target_indices=[0, 1],
        context_len=256,
        split_mode="chromosome",
        max_genes=40,
        skip_fasta_check=True,
        aux_pca_components=2,
        residual=True,
    )
    assert "y_final_t0" in manifest
    assert "y_final_t1" in manifest
    assert "aux_pca_0" in manifest
    assert "y_resid_final_t0" in manifest
    assert summary["sanity"]["max_abs_diff_by_layer"]["preds"] == 0.0


def test_v2_dataset_has_gene_mask(tmp_path):
    fasta = tmp_path / "toy.fa"
    fasta.write_text(">chr1\n" + "ACGT" * 300 + "\n")
    frame = pd.DataFrame(
        {
            "gene_id": ["g1"],
            "chrom": ["chr1"],
            "fasta_chrom": ["chr1"],
            "seq_start": [100],
            "seq_end": [356],
            "start": [150],
            "end": [250],
            "gene_start": [150],
            "gene_end": [250],
            "split": ["train"],
            "y_final_t0": [1.0],
        }
    )
    ds = DecimaV2GeneSequenceDataset(
        frame,
        fasta,
        label_columns={"final": ["y_final_t0"], "rep": [], "aux": [], "residual": [], "mid": []},
        normalizers={
            "final": {"mean": [0.0], "std": [1.0]},
            "rep": {"mean": [], "std": []},
            "aux": {"mean": [], "std": []},
            "residual": {"mean": [], "std": []},
            "mid": {"mean": [], "std": []},
        },
    )
    item = ds[0]
    assert item["x"].shape == (5, 256)
    assert item["x"][4].sum().item() == 100


def test_v2_tcn_forward_backward():
    model = build_v2_model(
        {
            "preset": "tcn_tiny",
            "channels": 8,
            "num_blocks": 2,
            "head_hidden": 16,
            "n_targets": 2,
            "n_replicates": 3,
            "n_aux": 4,
            "n_residual": 2,
            "n_mid": 5,
        }
    )
    x = torch.randn(2, 5, 1024)
    out = model(x)
    assert out["final"].shape == (2, 2)
    assert out["rep"].shape == (2, 2, 3)
    assert out["aux"].shape == (2, 4)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
