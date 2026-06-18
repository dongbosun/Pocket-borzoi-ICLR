from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def tiny_anndata() -> ad.AnnData:
    n_obs = 3
    n_vars = 66
    obs = pd.DataFrame(
        {
            "cell_type": ["astrocyte", "neuron", "microglia"],
            "tissue": ["brain", "brain", "brain"],
            "organ": ["brain", "brain", "brain"],
            "disease": ["healthy", "healthy", "control"],
            "n_cells": [100, 80, 50],
            "test_pearson": [0.9, 0.8, 0.7],
        }
    )
    rows = []
    for i in range(n_vars):
        chrom = f"chr{1 + (i % 22)}"
        tss = 1000 + i * 100
        rows.append(
            {
                "chrom": chrom,
                "start": tss - 50,
                "end": tss + 50,
                "gene_start": tss - 50,
                "gene_end": tss + 50,
                "strand": "+" if i % 2 == 0 else "-",
                "gene_type": "protein_coding",
                "gene_id": f"gene_{i}",
                "gene_name": f"G{i}",
                "gene_length": 100,
                "frac_N": 0.0,
                "frac_nan": 0.0,
                "mean_counts": float(i),
                "n_tracks": 1,
                "dataset": "toy",
                "fold": i % 5,
                "pearson": 0.5,
                "size_factor_pearson": 0.3,
                "ensembl_canonical_tss": tss,
            }
        )
    var = pd.DataFrame(rows)
    adata = ad.AnnData(X=np.zeros((n_obs, n_vars), dtype=np.float32), obs=obs, var=var)
    adata.layers["preds"] = np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    return adata
