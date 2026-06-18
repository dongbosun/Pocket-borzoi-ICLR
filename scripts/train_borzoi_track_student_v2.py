#!/usr/bin/env python
"""Train Pocket-Borzoi v2 multi-head track distillation student."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401

from pocketreg.borzoi.dataset_v2 import (  # noqa: E402
    BorzoiV2Dataset,
    compute_label_stats,
    load_v2_frame,
)
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.models.pocket_borzoi_v2 import PocketBorzoiV2  # noqa: E402
from pocketreg.models.small_cnn import count_parameters  # noqa: E402
from pocketreg.paths import checkpoints_dir, plots_dir, results_dir  # noqa: E402
from pocketreg.training.metrics import mae, pearsonr, r2_score, rmse  # noqa: E402
from pocketreg.training.utils import set_seed  # noqa: E402

LOGGER = logging.getLogger("train_borzoi_track_student_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--allow-local", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    try:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        spearman = float("nan")
    return {
        "pearson": pearsonr(y_true, y_pred),
        "spearman": spearman,
        "r2": r2_score(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def evaluate(model, loader, device, primary_mean: float, primary_std: float):
    import torch

    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(x)["primary"].detach().cpu().numpy()
            pred = pred * primary_std + primary_mean
            y_pred.extend(float(v) for v in pred)
            y_true.extend(float(v) for v in batch["primary_raw"])
            ids.extend(str(v) for v in batch["example_id"])
    return compute_metrics(y_true, y_pred), pd.DataFrame({"example_id": ids, "y_true": y_true, "y_pred": y_pred})


def save_plots(plot_dir: Path, history: list[dict[str, Any]], predictions: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    if not hist.empty:
        plt.figure(figsize=(6, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        plt.plot(hist["epoch"], hist["val_pearson"], label="val_pearson")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "train_curves.png", dpi=150)
        plt.close()
    for split, frame in predictions.items():
        if frame.empty:
            continue
        plt.figure(figsize=(4, 4))
        plt.scatter(frame["y_true"], frame["y_pred"], s=6, alpha=0.35)
        plt.xlabel("teacher primary q_mean")
        plt.ylabel("student pred")
        plt.tight_layout()
        plt.savefig(plot_dir / f"parity_{split}.png", dpi=150)
        plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.get("train", {})
    allow_local = bool(args.allow_local or train_cfg.get("allow_local", False))
    assert_compute_context("train_borzoi_track_student_v2", allow_local=allow_local, toy=False)
    print_cluster_context()
    set_seed(int(config.get("seed", 42)))

    import torch

    run_name = args.run_name or config.get("run_name", "k562_track_v2")
    run_dir = Path(config.get("output_dir", results_dir("runs"))) / run_name
    checkpoint_dir = Path(config.get("checkpoint_dir", checkpoints_dir(run_name)))
    plot_dir = Path(config.get("plot_dir", plots_dir(run_name)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    frame, label_cols = load_v2_frame(
        config["manifest_path"],
        config["rich_labels_path"],
        config["profile_pca_path"],
        config["aux_pca_path"],
        config["middle_projection_path"],
    )
    frame = frame[frame.get("status", "success") == "success"].copy()
    train_mask = frame["split"].astype(str).to_numpy() == "train"
    stats = compute_label_stats(frame, label_cols, train_mask)
    data_cfg = config.get("data", {})
    context_len = int(data_cfg.get("context_len", 131072))
    standardize = bool(data_cfg.get("target_standardize", True))
    max_rows = {
        "train": data_cfg.get("max_train_rows"),
        "val": data_cfg.get("max_val_rows"),
        "test": data_cfg.get("max_test_rows"),
    }
    datasets = {}
    for split in ("train", "val", "test"):
        sub = frame[frame["split"] == split].copy()
        datasets[split] = BorzoiV2Dataset(
            sub,
            label_cols,
            stats,
            fasta_path=config["fasta_path"],
            context_len=context_len,
            standardize=standardize,
            max_rows=int(max_rows[split]) if max_rows[split] is not None else None,
        )
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    loaders = {
        split: make_loader(ds, batch_size, split == "train", num_workers)
        for split, ds in datasets.items()
    }

    model_cfg = config.get("model", {})
    model = PocketBorzoiV2(
        input_channels=int(model_cfg.get("input_channels", 5)),
        channels=int(model_cfg.get("channels", 96)),
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        stem_stride=int(model_cfg.get("stem_stride", 16)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        head_hidden=int(model_cfg.get("head_hidden", 192)),
        fold_dim=len(label_cols["fold"]),
        profile_dim=len(label_cols["profile_pca"]),
        aux_dim=len(label_cols["aux_pca"]),
        middle_dim=len(label_cols["middle_proj"]),
    )
    requested_device = str(train_cfg.get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    max_epochs = int(train_cfg.get("max_epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))
    huber = torch.nn.HuberLoss(delta=float(train_cfg.get("huber_delta", 1.0)))
    mse = torch.nn.MSELoss()
    weights = config.get("loss_weights", {})
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 8))

    (run_dir / "model_summary.txt").write_text(
        f"{model}\n\nparameters_train={count_parameters(model)}\ncontext_len={context_len}\n"
        f"device={device}\nlabel_cols={json.dumps(label_cols, indent=2)}\n"
    )

    best_val = -np.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model(x)
            loss = weights.get("primary", 1.0) * huber(out["primary"], batch["primary"].to(device))
            loss = loss + weights.get("fold", 0.25) * huber(out["fold"], batch["fold"].to(device, dtype=torch.float32))
            loss = loss + weights.get("profile_pca", 0.2) * mse(out["profile_pca"], batch["profile_pca"].to(device, dtype=torch.float32))
            loss = loss + weights.get("aux_pca", 0.1) * mse(out["aux_pca"], batch["aux_pca"].to(device, dtype=torch.float32))
            loss = loss + weights.get("middle_proj", 0.05) * mse(out["middle_proj"], batch["middle_proj"].to(device, dtype=torch.float32))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        primary_mean = float(stats["primary"].mean[0])
        primary_std = float(stats["primary"].std[0])
        val_metrics, _ = evaluate(model, loaders["val"], device, primary_mean, primary_std)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
        }
        history.append(rec)
        LOGGER.info("epoch=%s train_loss=%.5f val_pearson=%.4f", epoch, rec["train_loss"], rec["val_pearson"])
        if np.isfinite(rec["val_pearson"]) and rec["val_pearson"] > best_val:
            best_val = rec["val_pearson"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stats": {k: {"mean": v.mean.tolist(), "std": v.std.tolist()} for k, v in stats.items()},
                    "label_cols": label_cols,
                    "context_len": context_len,
                },
                checkpoint_dir / "model_best.pt",
            )
            torch.save(
                {
                    "inference_state_dict": model.inference_state_dict(),
                    "config": config,
                    "primary_mean": primary_mean,
                    "primary_std": primary_std,
                    "context_len": context_len,
                },
                checkpoint_dir / "model_inference_only.pt",
            )
        if epoch - best_epoch >= patience:
            LOGGER.info("Early stopping at epoch %s; best_epoch=%s", epoch, best_epoch)
            break

    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_dir / "model_last.pt")
    if (checkpoint_dir / "model_best.pt").exists():
        ckpt = torch.load(checkpoint_dir / "model_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    primary_mean = float(stats["primary"].mean[0])
    primary_std = float(stats["primary"].std[0])
    metrics: dict[str, Any] = {
        "run_name": run_name,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_pearson": best_val,
        "runtime_seconds": time.time() - started,
        "checkpoint_dir": str(checkpoint_dir),
        "plot_dir": str(plot_dir),
        "num_rows": {split: len(ds) for split, ds in datasets.items()},
        "params_train": count_parameters(model),
    }
    predictions = {}
    for split, loader in loaders.items():
        split_metrics, pred = evaluate(model, loader, device, primary_mean, primary_std)
        metrics[split] = split_metrics
        predictions[split] = pred
        pred.to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
    pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    save_plots(plot_dir, history, predictions)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
