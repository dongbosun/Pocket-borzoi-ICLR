"""Plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_parity_plot(y_true, y_pred, out_path: str | Path, title: str = "parity") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred, s=8, alpha=0.6)
    ax.set_xlabel("teacher")
    ax.set_ylabel("student")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
