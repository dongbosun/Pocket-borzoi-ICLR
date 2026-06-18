"""End-to-end training and prediction loops."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pocketreg.data.manifest import DecimaGeneSequenceDataset
from pocketreg.eval.plots import (
    plot_parity,
    plot_prediction_distribution,
    plot_residuals,
    plot_training_curves,
)
from pocketreg.models.small_cnn import build_model, model_summary
from pocketreg.training.losses import build_loss
from pocketreg.training.metrics import regression_metrics
from pocketreg.training.utils import resolve_device, save_json, save_yaml, set_seed

LOGGER = logging.getLogger(__name__)
BASELINE_NUMERIC_COLS = [
    "gene_length",
    "frac_N",
    "frac_nan",
    "mean_counts",
    "n_tracks",
    "pearson",
    "size_factor_pearson",
]


def _split_manifest(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = manifest.loc[manifest["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"Manifest has no rows for split={split!r}.")
    return frame.reset_index(drop=True)


def _make_loader(
    frame: pd.DataFrame,
    fasta_path: str | Path,
    y_mean: float,
    y_std: float,
    train_cfg: dict[str, Any],
    *,
    shuffle: bool,
    augment_rc: bool = False,
) -> DataLoader:
    dataset = DecimaGeneSequenceDataset(
        frame,
        fasta_path,
        y_mean,
        y_std,
        augment_rc=augment_rc,
        cache_size=int(train_cfg.get("cache_size", 512)),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    *,
    train_mean_raw: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run model prediction for one DataLoader and return predictions plus metrics."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            pred_std = model(x).detach().cpu().numpy().astype(np.float64)
            y_true = batch["y_raw"].detach().cpu().numpy().astype(np.float64)
            pred_raw = pred_std * float(y_std) + float(y_mean)
            for i in range(len(y_true)):
                rows.append(
                    {
                        "gene_id": batch["gene_id"][i],
                        "chrom": batch["chrom"][i],
                        "split": batch["split"][i],
                        "y_teacher": float(y_true[i]),
                        "y_pred": float(pred_raw[i]),
                        "y_pred_standardized": float(pred_std[i]),
                    }
                )
    pred_df = pd.DataFrame(rows)
    metrics = regression_metrics(
        pred_df["y_teacher"].to_numpy(),
        pred_df["y_pred"].to_numpy(),
        train_mean=train_mean_raw,
    )
    return pred_df, metrics


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    grad_clip_norm: float | None,
    amp: bool,
) -> float:
    model.train()
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            pred = model(x)
            loss = criterion(pred, y)
        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _copy_sidecars(manifest_path: Path, output_dir: Path) -> None:
    for name in ("target_metadata.json", "manifest_summary.json"):
        src = manifest_path.parent / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    y_mean: float,
    y_std: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": config["model"],
        "train_config": config.get("train", {}),
        "context_len": config.get("context_len"),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "epoch": int(epoch),
        "metrics": metrics,
    }


def _save_predictions_and_plots(
    predictions: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    train_log: pd.DataFrame,
    output_dir: Path,
    save_plots: bool,
) -> None:
    for split, pred_df in predictions.items():
        pred_df.to_parquet(output_dir / f"predictions_{split}.parquet", index=False)
    if not save_plots:
        return
    plot_dir = output_dir / "plots"
    plot_training_curves(train_log, plot_dir / "train_curves.png")
    for split in ("val", "test"):
        if split not in predictions:
            continue
        pred_df = predictions[split]
        y_true = pred_df["y_teacher"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        plot_parity(y_true, y_pred, metrics[split], plot_dir / f"parity_{split}.png", split)
        if split == "test":
            plot_residuals(y_true, y_pred, plot_dir / "residuals_test.png", "test residuals")
            plot_prediction_distribution(
                y_true,
                y_pred,
                plot_dir / "pred_distribution_test.png",
                "test prediction distribution",
            )


def fit_baselines(manifest: pd.DataFrame) -> dict[str, Any]:
    """Fit mean and metadata Ridge baselines on train rows."""
    train = _split_manifest(manifest, "train")
    train_mean = float(train["y_teacher"].mean())
    out: dict[str, Any] = {"mean": {}, "metadata_ridge": {}}
    for split in ("train", "val", "test"):
        frame = _split_manifest(manifest, split)
        pred = np.full(len(frame), train_mean)
        out["mean"][split] = regression_metrics(frame["y_teacher"].to_numpy(), pred, train_mean=train_mean)

    numeric_cols = [col for col in BASELINE_NUMERIC_COLS if col in manifest]
    if not numeric_cols:
        out["metadata_ridge"]["error"] = "No supported numeric gene metadata columns were found."
        return out
    try:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        model.fit(train[numeric_cols], train["y_teacher"])
        out["metadata_ridge"]["features"] = numeric_cols
        for split in ("train", "val", "test"):
            frame = _split_manifest(manifest, split)
            pred = model.predict(frame[numeric_cols])
            out["metadata_ridge"][split] = regression_metrics(
                frame["y_teacher"].to_numpy(), pred, train_mean=train_mean
            )
    except Exception as exc:
        out["metadata_ridge"]["error"] = str(exc)
    return out


def train_from_config(
    config: dict[str, Any],
    *,
    run_name: str | None = None,
    manifest_path: str | Path | None = None,
    fasta_path: str | Path | None = None,
) -> Path:
    """Train a Decima student model from a config dictionary."""
    config = dict(config)
    if manifest_path is not None:
        config["manifest_path"] = str(manifest_path)
    if fasta_path is not None:
        config["fasta_path"] = str(fasta_path)
    manifest_path = Path(config["manifest_path"])
    fasta_path = Path(config["fasta_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    output_dir = Path(config.get("output_dir", "outputs/runs/decima_mvp"))
    if run_name:
        output_dir = output_dir.parent / run_name
        config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    seed = int(config.get("seed", 13))
    set_seed(seed)
    train_cfg = config.get("train", {})
    logging_cfg = config.get("logging", {})
    device = resolve_device(str(train_cfg.get("device", "auto")))
    LOGGER.info("Training on device=%s", device)

    manifest = pd.read_parquet(manifest_path)
    train_frame = _split_manifest(manifest, "train")
    val_frame = _split_manifest(manifest, "val")
    test_frame = _split_manifest(manifest, "test")
    target_standardize = bool(train_cfg.get("target_standardize", True))
    if target_standardize:
        y_mean = float(train_frame["y_teacher"].mean())
        y_std = float(train_frame["y_teacher"].std(ddof=0))
        if y_std <= 0 or not np.isfinite(y_std):
            y_std = 1.0
    else:
        y_mean = 0.0
        y_std = 1.0
    train_mean_raw = float(train_frame["y_teacher"].mean())

    train_loader = _make_loader(
        train_frame,
        fasta_path,
        y_mean,
        y_std,
        train_cfg,
        shuffle=True,
        augment_rc=bool(train_cfg.get("augment_rc", False)),
    )
    eval_cfg = dict(train_cfg)
    eval_cfg["num_workers"] = int(train_cfg.get("eval_num_workers", train_cfg.get("num_workers", 0)))
    val_loader = _make_loader(val_frame, fasta_path, y_mean, y_std, eval_cfg, shuffle=False)
    test_loader = _make_loader(test_frame, fasta_path, y_mean, y_std, eval_cfg, shuffle=False)

    model = build_model(config["model"]).to(device)
    (output_dir / "model_summary.txt").write_text(model_summary(model))
    save_yaml(config, output_dir / "config.yaml")
    _copy_sidecars(manifest_path, output_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = None
    if str(train_cfg.get("scheduler", "cosine")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(train_cfg.get("max_epochs", 50))
        )
    criterion = build_loss(str(train_cfg.get("loss", "huber")), float(train_cfg.get("huber_delta", 1.0)))
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"

    train_log_rows = []
    best_score = -np.inf
    best_epoch = -1
    patience = int(train_cfg.get("early_stopping_patience", 8))
    max_epochs = int(train_cfg.get("max_epochs", 50))
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            amp=amp,
        )
        if scheduler is not None:
            scheduler.step()
        val_pred, val_metrics = predict_loader(
            model, val_loader, device, y_mean, y_std, train_mean_raw=train_mean_raw
        )
        val_loss = float(np.mean((val_pred["y_pred"] - val_pred["y_teacher"]) ** 2))
        score = val_metrics.get("pearson")
        if score is None or not np.isfinite(score):
            score = -float(val_metrics.get("rmse", np.inf))
        train_log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pearson": val_metrics.get("pearson"),
                "val_rmse": val_metrics.get("rmse"),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        LOGGER.info(
            "epoch=%s train_loss=%.4f val_pearson=%s val_rmse=%.4f",
            epoch,
            train_loss,
            val_metrics.get("pearson"),
            val_metrics.get("rmse", float("nan")),
        )
        payload = _checkpoint_payload(model, config, epoch, y_mean, y_std, {"val": val_metrics})
        torch.save(payload, output_dir / "checkpoints" / "last.pt")
        if float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save(payload, output_dir / "checkpoints" / "best.pt")
        if epoch - best_epoch >= patience:
            LOGGER.info("Early stopping at epoch %s; best epoch was %s.", epoch, best_epoch)
            break

    checkpoint = torch.load(output_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for split, loader in {"train": train_loader, "val": val_loader, "test": test_loader}.items():
        pred_df, split_metrics = predict_loader(
            model, loader, device, y_mean, y_std, train_mean_raw=train_mean_raw
        )
        predictions[split] = pred_df
        metrics[split] = split_metrics
    metrics["best_epoch"] = {"epoch": best_epoch, "score": best_score}
    train_log = pd.DataFrame(train_log_rows)
    train_log.to_csv(output_dir / "train_log.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    save_json(fit_baselines(manifest), output_dir / "metrics_baselines.json")
    _save_predictions_and_plots(
        predictions,
        metrics,
        train_log,
        output_dir,
        save_plots=bool(logging_cfg.get("save_plots", True)),
    )
    return output_dir


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpoint and rebuild its model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    fasta_path: str | Path,
    out_dir: str | Path,
    *,
    device_name: str = "auto",
    split: str | None = None,
    batch_size: int | None = None,
) -> Path:
    """Evaluate a saved checkpoint and write predictions, metrics, and plots."""
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    manifest = pd.read_parquet(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_mean = float(checkpoint.get("y_mean", 0.0))
    y_std = float(checkpoint.get("y_std", 1.0))
    train_mean_raw = float(_split_manifest(manifest, "train")["y_teacher"].mean())
    train_cfg = dict(checkpoint.get("train_config", {}))
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size
    train_cfg["num_workers"] = int(train_cfg.get("eval_num_workers", 0))
    selected_splits = [split] if split else ["train", "val", "test"]

    predictions = {}
    metrics = {}
    for split_name in selected_splits:
        frame = _split_manifest(manifest, split_name)
        loader = _make_loader(frame, fasta_path, y_mean, y_std, train_cfg, shuffle=False)
        pred_df, split_metrics = predict_loader(
            model, loader, device, y_mean, y_std, train_mean_raw=train_mean_raw
        )
        pred_df.to_parquet(out_dir / f"predictions_{split_name}.parquet", index=False)
        predictions[split_name] = pred_df
        metrics[split_name] = split_metrics
    save_json(metrics, out_dir / "metrics.json")
    plot_dir = out_dir / "plots"
    for split_name, pred_df in predictions.items():
        y_true = pred_df["y_teacher"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()
        plot_parity(
            y_true,
            y_pred,
            metrics[split_name],
            plot_dir / f"parity_{split_name}.png",
            split_name,
        )
        if split_name == "test":
            plot_residuals(y_true, y_pred, plot_dir / "residuals_test.png", "test residuals")
            plot_prediction_distribution(
                y_true,
                y_pred,
                plot_dir / "pred_distribution_test.png",
                "test prediction distribution",
            )
    return out_dir
