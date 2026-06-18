#!/usr/bin/env python
"""Train a small CNN student on Borzoi K562 gene-level teacher labels."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import _path  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.data.fasta import FastaReader  # noqa: E402
from pocketreg.data.sequence import one_hot_encode  # noqa: E402
from pocketreg.models.small_cnn import SmallCNN, count_parameters  # noqa: E402
from pocketreg.paths import checkpoints_dir, plots_dir, results_dir  # noqa: E402
from pocketreg.training.metrics import mae, pearsonr, r2_score, rmse  # noqa: E402
from pocketreg.training.utils import set_seed  # noqa: E402

LOGGER = logging.getLogger("train_borzoi_track_student")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


class TrackDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        fasta_path: Path,
        context_len: int,
        target_mean: float,
        target_std: float,
        standardize: bool,
    ):
        self.rows = rows
        self.fasta_path = Path(fasta_path)
        self._fasta: FastaReader | None = None
        self.context_len = int(context_len)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if target_std > 0 else 1.0
        self.standardize = standardize

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def fasta(self) -> FastaReader:
        if self._fasta is None:
            self._fasta = FastaReader(self.fasta_path)
        return self._fasta

    def __getitem__(self, index: int):
        row = self.rows[index]
        tss = int(row["tss"])
        start = tss - self.context_len // 2
        end = start + self.context_len
        seq = self.fasta.fetch(str(row["chrom"]), start, end, pad=True)
        x = one_hot_encode(seq, channels_first=True)
        y = float(row["q_teacher"])
        y_std = (y - self.target_mean) / self.target_std if self.standardize else y
        import torch

        return (
            torch.from_numpy(x),
            torch.tensor(y_std, dtype=torch.float32),
            row["example_id"],
            y,
        )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def split_rows(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    rows = {}
    for split in ("train", "val", "test"):
        sub = df[(df["split"] == split) & (df["status"] == "success")].copy()
        rows[split] = sub.to_dict(orient="records")
    return rows


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


def evaluate(model, loader, device, target_mean: float, target_std: float, standardize: bool):
    import torch

    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    example_ids: list[str] = []
    with torch.no_grad():
        for x, _, ids, y_raw in loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            pred = model(x).detach().cpu().numpy()
            if standardize:
                pred = pred * target_std + target_mean
            y_pred.extend(float(v) for v in pred)
            y_true.extend(float(v) for v in y_raw)
            example_ids.extend(str(v) for v in ids)
    return compute_metrics(y_true, y_pred), pd.DataFrame(
        {"example_id": example_ids, "y_true": y_true, "y_pred": y_pred}
    )


def save_plots(plot_dir: Path, history: list[dict[str, Any]], predictions: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("Skipping plots: %s", exc)
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    if history:
        hist = pd.DataFrame(history)
        plt.figure(figsize=(6, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        if "val_pearson" in hist:
            plt.plot(hist["epoch"], hist["val_pearson"], label="val_pearson")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "train_curves.png", dpi=150)
        plt.close()
    for split, frame in predictions.items():
        if frame.empty:
            continue
        plt.figure(figsize=(4, 4))
        plt.scatter(frame["y_true"], frame["y_pred"], s=8, alpha=0.5)
        lo = min(frame["y_true"].min(), frame["y_pred"].min())
        hi = max(frame["y_true"].max(), frame["y_pred"].max())
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.xlabel("teacher q")
        plt.ylabel("student pred")
        plt.tight_layout()
        plt.savefig(plot_dir / f"parity_{split}.png", dpi=150)
        plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.get("train", {})
    toy = bool(args.toy or config.get("toy", False))
    allow_local = bool(args.allow_local or train_cfg.get("allow_local", False))
    assert_compute_context("train_borzoi_track_student", allow_local=allow_local, toy=toy)
    print_cluster_context()
    set_seed(int(config.get("seed", 42)))

    import torch

    labels_path = args.labels or Path(config.get("labels_path"))
    fasta_path = args.fasta or Path(config.get("fasta_path"))
    run_name = args.run_name or config.get("run_name", "k562_track_student")
    output_dir = Path(config.get("output_dir", results_dir("runs")))
    run_dir = output_dir / run_name
    checkpoint_dir = Path(config.get("checkpoint_dir", checkpoints_dir(run_name)))
    plot_dir = Path(config.get("plot_dir", plots_dir(run_name)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    df = pd.read_parquet(labels_path)
    if "status" not in df:
        df["status"] = "success"
    rows_by_split = split_rows(df)
    if not rows_by_split["train"] or not rows_by_split["val"]:
        raise ValueError("Need non-empty train and val splits for training")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    context_len = int(data_cfg.get("context_len", 65536))
    standardize = bool(data_cfg.get("target_standardize", True))
    train_y = np.array([float(row["q_teacher"]) for row in rows_by_split["train"]], dtype=float)
    target_mean = float(np.mean(train_y))
    target_std = float(np.std(train_y))
    if target_std <= 0:
        target_std = 1.0

    datasets = {
        split: TrackDataset(
            rows,
            fasta_path=fasta_path,
            context_len=context_len,
            target_mean=target_mean,
            target_std=target_std,
            standardize=standardize,
        )
        for split, rows in rows_by_split.items()
    }
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    loaders = {
        "train": make_loader(datasets["train"], batch_size, True, num_workers),
        "val": make_loader(datasets["val"], batch_size, False, num_workers),
        "test": make_loader(datasets["test"], batch_size, False, num_workers),
    }

    model = SmallCNN(
        channels=int(model_cfg.get("channels", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 6)),
        stem_stride=int(model_cfg.get("stem_stride", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        head_hidden=int(model_cfg.get("head_hidden", 128)),
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
    loss_fn = torch.nn.HuberLoss(delta=float(train_cfg.get("huber_delta", 1.0)))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 8))

    (run_dir / "model_summary.txt").write_text(
        f"{model}\n\nparameters={count_parameters(model)}\ncontext_len={context_len}\n"
        f"device={device}\ntarget_mean={target_mean}\ntarget_std={target_std}\n"
    )

    best_val = -np.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y, _, _ in loaders["train"]:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_metrics, _ = evaluate(
            model,
            loaders["val"],
            device,
            target_mean=target_mean,
            target_std=target_std,
            standardize=standardize,
        )
        train_loss = float(np.mean(losses)) if losses else float("nan")
        record = {"epoch": epoch, "train_loss": train_loss, "val_pearson": val_metrics["pearson"]}
        history.append(record)
        LOGGER.info("epoch=%s train_loss=%.5f val_pearson=%.4f", epoch, train_loss, val_metrics["pearson"])
        if np.isfinite(val_metrics["pearson"]) and val_metrics["pearson"] > best_val:
            best_val = val_metrics["pearson"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "context_len": context_len,
                    "model_cfg": dict(model_cfg),
                },
                checkpoint_dir / "checkpoint_best.pt",
            )
        if epoch - best_epoch >= patience:
            LOGGER.info("Early stopping at epoch %s; best_epoch=%s", epoch, best_epoch)
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "target_mean": target_mean,
            "target_std": target_std,
            "context_len": context_len,
            "model_cfg": dict(model_cfg),
        },
        checkpoint_dir / "checkpoint_last.pt",
    )
    if (checkpoint_dir / "checkpoint_best.pt").exists():
        checkpoint = torch.load(checkpoint_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    metrics: dict[str, Any] = {
        "run_name": run_name,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_pearson": best_val,
        "runtime_seconds": time.time() - started,
        "checkpoint_dir": str(checkpoint_dir),
        "plot_dir": str(plot_dir),
        "num_rows": {split: len(rows) for split, rows in rows_by_split.items()},
    }
    predictions: dict[str, pd.DataFrame] = {}
    for split, loader in loaders.items():
        split_metrics, frame = evaluate(
            model,
            loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
            standardize=standardize,
        )
        metrics[split] = split_metrics
        predictions[split] = frame
        frame.to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
    pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    save_plots(plot_dir, history, predictions)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
