"""Matplotlib plots for distillation runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(train_log: pd.DataFrame, out_path: str | Path) -> None:
    """Save train/validation loss and validation Pearson curves."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(train_log["epoch"], train_log["train_loss"], label="train")
    if "val_loss" in train_log:
        axes[0].plot(train_log["epoch"], train_log["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    if "val_pearson" in train_log:
        axes[1].plot(train_log["epoch"], train_log["val_pearson"])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val Pearson")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    out_path: str | Path,
    title: str,
) -> None:
    """Save a teacher-vs-student parity plot."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.55)
    lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Decima teacher")
    ax.set_ylabel("Student prediction")
    subtitle = (
        f"Pearson={metrics.get('pearson', float('nan')):.3f}, "
        f"Spearman={metrics.get('spearman', float('nan')):.3f}, "
        f"R2={metrics.get('r2', float('nan')):.3f}"
    )
    ax.set_title(f"{title}\n{subtitle}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str) -> None:
    """Save residuals against teacher labels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred - y_true, s=8, alpha=0.55)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Decima teacher")
    ax.set_ylabel("Student - teacher")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_prediction_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str
) -> None:
    """Save overlaid histograms of teacher and student predictions."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_true, bins=40, alpha=0.55, label="teacher")
    ax.hist(y_pred, bins=40, alpha=0.55, label="student")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
