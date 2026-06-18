"""Training loop for Pocket-Decima targeted distillation v2."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pocketreg.data.v2 import DecimaV2CachedGeneSequenceDataset, DecimaV2GeneSequenceDataset
from pocketreg.eval.plots import (
    plot_parity,
    plot_prediction_distribution,
    plot_residuals,
    plot_training_curves,
)
from pocketreg.models.tcn import build_v2_model, v2_model_summary
from pocketreg.training.metrics import regression_metrics
from pocketreg.training.utils import resolve_device, save_json, save_yaml, set_seed

LOGGER = logging.getLogger(__name__)
FINAL_RE = re.compile(r"^y_final_t(\d+)$")
RESIDUAL_RE = re.compile(r"^y_resid_final_t(\d+)$")


def _split_manifest(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    frame = manifest.loc[manifest["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"Manifest has no rows for split={split!r}.")
    return frame.reset_index(drop=True)


def _suffix_int(name: str) -> int:
    match = re.search(r"_t(\d+)$", name)
    return int(match.group(1)) if match else -1


def infer_label_columns(manifest: pd.DataFrame) -> dict[str, list[str]]:
    """Infer v2 label column groups from a manifest."""
    final = sorted([c for c in manifest.columns if FINAL_RE.match(c)], key=_suffix_int)
    rep = []
    for layer_idx in range(4):
        layer_cols = sorted(
            [c for c in manifest.columns if c.startswith(f"y_rep_v1_rep{layer_idx}_t")],
            key=_suffix_int,
        )
        rep.extend(layer_cols)
    aux = sorted([c for c in manifest.columns if c.startswith("aux_pca_")], key=lambda c: int(c.rsplit("_", 1)[1]))
    residual = sorted([c for c in manifest.columns if RESIDUAL_RE.match(c)], key=_suffix_int)
    mid = sorted([c for c in manifest.columns if c.startswith("mid_")])
    if not final:
        raise ValueError("v2 manifest has no y_final_t* columns.")
    return {"final": final, "rep": rep, "aux": aux, "residual": residual, "mid": mid}


def _normalizers(manifest: pd.DataFrame, label_columns: dict[str, list[str]]) -> dict[str, dict[str, list[float]]]:
    train = _split_manifest(manifest, "train")
    out: dict[str, dict[str, list[float]]] = {}
    for group, cols in label_columns.items():
        if not cols:
            out[group] = {"mean": [], "std": []}
            continue
        values = train[cols].astype(float).to_numpy()
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        out[group] = {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        }
    return out


def _make_loader(
    frame: pd.DataFrame,
    fasta_path: str | Path,
    label_columns: dict[str, list[str]],
    normalizers: dict[str, dict[str, list[float]]],
    train_cfg: dict[str, Any],
    *,
    shuffle: bool,
    augment_rc: bool = False,
    input_channels: int = 5,
) -> DataLoader:
    sequence_cache_dir = train_cfg.get("sequence_cache_dir")
    if sequence_cache_dir:
        dataset = DecimaV2CachedGeneSequenceDataset(
            frame,
            sequence_cache_dir,
            label_columns=label_columns,
            normalizers=normalizers,
            augment_rc=augment_rc,
            input_channels=int(input_channels),
        )
    else:
        dataset = DecimaV2GeneSequenceDataset(
            frame,
            fasta_path,
            label_columns=label_columns,
            normalizers=normalizers,
            augment_rc=augment_rc,
            cache_size=int(train_cfg.get("cache_size", 512)),
            input_channels=int(input_channels),
        )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _build_criterion(name: str, huber_delta: float) -> torch.nn.Module:
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "huber":
        return torch.nn.SmoothL1Loss(beta=float(huber_delta))
    raise ValueError("loss must be mse or huber.")


def _rep_prediction_flat(pred: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "rep" not in pred:
        return None
    # Manifest order is v1_rep0_t0, v1_rep0_t1, v1_rep1_t0, ...
    return pred["rep"].permute(0, 2, 1).reshape(pred["rep"].shape[0], -1)


def _batch_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, Any],
    criterion: torch.nn.Module,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}
    if batch["final"].numel() and weights.get("final", 1.0) > 0:
        losses["final"] = criterion(pred["final"], batch["final"])
    rep_pred = _rep_prediction_flat(pred)
    if rep_pred is not None and batch["rep"].numel() and weights.get("rep", 0.0) > 0:
        losses["rep"] = criterion(rep_pred, batch["rep"])
    for group in ("aux", "residual", "mid"):
        if group in pred and batch[group].numel() and weights.get(group, 0.0) > 0:
            losses[group] = criterion(pred[group], batch[group])
    if not losses:
        raise ValueError("No active v2 losses; check label columns and loss weights.")
    total = sum(float(weights.get(k, 1.0)) * v for k, v in losses.items())
    return total, {k: float(v.detach().cpu()) for k, v in losses.items()}


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    weights: dict[str, float],
    *,
    grad_clip_norm: float | None,
    amp: bool,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_losses = []
    group_losses: dict[str, list[float]] = {}
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for batch in tqdm(loader, desc="train-v2", leave=False):
        optimizer.zero_grad(set_to_none=True)
        for key in ("x", "final", "rep", "aux", "residual", "mid"):
            batch[key] = batch[key].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            pred = model(batch["x"])
            loss, parts = _batch_loss(pred, batch, criterion, weights)
        scaler.scale(loss).backward()
        if grad_clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        total_losses.append(float(loss.detach().cpu()))
        for name, value in parts.items():
            group_losses.setdefault(name, []).append(value)
    return float(np.mean(total_losses)), {k: float(np.mean(v)) for k, v in group_losses.items()}


def _unstandardize(values: np.ndarray, normalizer: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    return values.astype(np.float64) * std + mean


def predict_v2_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_columns: dict[str, list[str]],
    normalizers: dict[str, dict[str, list[float]]],
    *,
    train_mean_raw: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict final target outputs and return a DataFrame plus metrics."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            pred_std = model(x)["final"].detach().cpu().numpy()
            pred_raw = _unstandardize(pred_std, normalizers["final"])
            true_raw = batch["final_raw"].detach().cpu().numpy().astype(np.float64)
            for i in range(pred_raw.shape[0]):
                row = {
                    "gene_id": batch["gene_id"][i],
                    "chrom": batch["chrom"][i],
                    "split": batch["split"][i],
                }
                for j, col in enumerate(label_columns["final"]):
                    row[f"{col}_true"] = float(true_raw[i, j])
                    row[f"{col}_pred"] = float(pred_raw[i, j])
                rows.append(row)
    pred_df = pd.DataFrame(rows)
    metrics: dict[str, Any] = {}
    for j, col in enumerate(label_columns["final"]):
        split_metrics = regression_metrics(
            pred_df[f"{col}_true"].to_numpy(),
            pred_df[f"{col}_pred"].to_numpy(),
            train_mean=train_mean_raw if j == 0 else None,
        )
        metrics[col] = split_metrics
        if j == 0:
            metrics.update(split_metrics)
    return pred_df, metrics


def _copy_sidecars(manifest_path: Path, output_dir: Path) -> None:
    for name in ("target_metadata.json", "manifest_summary.json", "v2_signal_summary.json"):
        src = manifest_path.parent / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    label_columns: dict[str, list[str]],
    normalizers: dict[str, dict[str, list[float]]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": config["model"],
        "train_config": config.get("train", {}),
        "context_len": config.get("context_len"),
        "epoch": int(epoch),
        "label_columns": label_columns,
        "normalizers": normalizers,
        "metrics": metrics,
        "version": "pocket-decima-targeted-distillation-v2",
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
        y_true = pred_df["y_final_t0_true"].to_numpy()
        y_pred = pred_df["y_final_t0_pred"].to_numpy()
        plot_parity(y_true, y_pred, metrics[split], plot_dir / f"parity_{split}.png", split)
        if split == "test":
            plot_residuals(y_true, y_pred, plot_dir / "residuals_test.png", "test residuals")
            plot_prediction_distribution(
                y_true,
                y_pred,
                plot_dir / "pred_distribution_test.png",
                "test prediction distribution",
            )


def train_v2_from_config(
    config: dict[str, Any],
    *,
    run_name: str | None = None,
    manifest_path: str | Path | None = None,
    fasta_path: str | Path | None = None,
) -> Path:
    """Train a Pocket-Decima v2 targeted distillation model."""
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

    output_dir = Path(config.get("output_dir", "outputs/runs/decima_v2"))
    if run_name:
        output_dir = output_dir.parent / run_name
        config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    seed = int(config.get("seed", 13))
    set_seed(seed)
    train_cfg = config.get("train", {})
    if config.get("sequence_cache_dir") and not train_cfg.get("sequence_cache_dir"):
        train_cfg = dict(train_cfg)
        train_cfg["sequence_cache_dir"] = config["sequence_cache_dir"]
    logging_cfg = config.get("logging", {})
    device = resolve_device(str(train_cfg.get("device", "auto")))
    LOGGER.info("Training v2 on device=%s", device)

    manifest = pd.read_parquet(manifest_path)
    label_columns = infer_label_columns(manifest)
    if float(train_cfg.get("rep_loss_weight", 0.25)) <= 0:
        label_columns["rep"] = []
    if float(train_cfg.get("aux_loss_weight", 0.1)) <= 0:
        label_columns["aux"] = []
    if float(train_cfg.get("residual_loss_weight", 0.2)) <= 0:
        label_columns["residual"] = []
    if float(train_cfg.get("mid_loss_weight", 0.1)) <= 0:
        label_columns["mid"] = []
    normalizers = _normalizers(manifest, label_columns)
    n_targets = len(label_columns["final"])
    n_replicates = len(label_columns["rep"]) // max(n_targets, 1)
    config["model"] = dict(config["model"])
    input_channels = int(config["model"].get("input_channels", 5))
    config["model"].update(
        {
            "input_channels": input_channels,
            "n_targets": n_targets,
            "n_replicates": n_replicates,
            "n_aux": len(label_columns["aux"]),
            "n_mid": len(label_columns["mid"]),
            "n_residual": len(label_columns["residual"]),
        }
    )
    train_frame = _split_manifest(manifest, "train")
    val_frame = _split_manifest(manifest, "val")
    test_frame = _split_manifest(manifest, "test")
    train_mean_raw = float(train_frame["y_final_t0"].mean())
    eval_cfg = dict(train_cfg)
    eval_cfg["num_workers"] = int(train_cfg.get("eval_num_workers", train_cfg.get("num_workers", 0)))

    train_loader = _make_loader(
        train_frame,
        fasta_path,
        label_columns,
        normalizers,
        train_cfg,
        shuffle=True,
        augment_rc=bool(train_cfg.get("augment_rc", False)),
        input_channels=input_channels,
    )
    val_loader = _make_loader(
        val_frame, fasta_path, label_columns, normalizers, eval_cfg, shuffle=False, input_channels=input_channels
    )
    test_loader = _make_loader(
        test_frame, fasta_path, label_columns, normalizers, eval_cfg, shuffle=False, input_channels=input_channels
    )

    model = build_v2_model(config["model"]).to(device)
    (output_dir / "model_summary.txt").write_text(v2_model_summary(model))
    save_yaml(config, output_dir / "config.yaml")
    save_json({"label_columns": label_columns, "normalizers": normalizers}, output_dir / "label_spec.json")
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
    criterion = _build_criterion(str(train_cfg.get("loss", "huber")), float(train_cfg.get("huber_delta", 1.0)))
    weights = {
        "final": float(train_cfg.get("final_loss_weight", 1.0)),
        "rep": float(train_cfg.get("rep_loss_weight", 0.25)),
        "aux": float(train_cfg.get("aux_loss_weight", 0.1)),
        "mid": float(train_cfg.get("mid_loss_weight", 0.1)),
        "residual": float(train_cfg.get("residual_loss_weight", 0.2)),
    }
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    patience = int(train_cfg.get("early_stopping_patience", 8))
    max_epochs = int(train_cfg.get("max_epochs", 50))
    best_score = -np.inf
    best_epoch = -1
    train_log_rows = []

    for epoch in range(1, max_epochs + 1):
        train_loss, group_losses = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            weights,
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            amp=amp,
        )
        if scheduler is not None:
            scheduler.step()
        val_pred, val_metrics = predict_v2_loader(
            model, val_loader, device, label_columns, normalizers, train_mean_raw=train_mean_raw
        )
        val_loss = float(np.mean((val_pred["y_final_t0_pred"] - val_pred["y_final_t0_true"]) ** 2))
        score = val_metrics.get("pearson")
        if score is None or not np.isfinite(score):
            score = -float(val_metrics.get("rmse", np.inf))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pearson": val_metrics.get("pearson"),
            "val_rmse": val_metrics.get("rmse"),
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update({f"train_{k}_loss": v for k, v in group_losses.items()})
        train_log_rows.append(row)
        LOGGER.info(
            "epoch=%s train_loss=%.4f val_pearson=%s val_rmse=%.4f",
            epoch,
            train_loss,
            val_metrics.get("pearson"),
            val_metrics.get("rmse", float("nan")),
        )
        payload = _checkpoint_payload(model, config, epoch, label_columns, normalizers, {"val": val_metrics})
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
        pred_df, split_metrics = predict_v2_loader(
            model, loader, device, label_columns, normalizers, train_mean_raw=train_mean_raw
        )
        predictions[split] = pred_df
        metrics[split] = split_metrics
    metrics["best_epoch"] = {"epoch": best_epoch, "score": best_score}
    train_log = pd.DataFrame(train_log_rows)
    train_log.to_csv(output_dir / "train_log.csv", index=False)
    save_json(metrics, output_dir / "metrics.json")
    _save_predictions_and_plots(
        predictions,
        metrics,
        train_log,
        output_dir,
        save_plots=bool(logging_cfg.get("save_plots", True)),
    )
    return output_dir
