#!/usr/bin/env python
"""Train Borzoi delta v2 with asinh targets and effect-balanced sampling."""

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

from pocketreg.borzoi.delta_dataset_v2 import (  # noqa: E402
    DeltaV2Dataset,
    build_delta_metadata,
    compute_sample_weights,
    estimate_delta_scale,
    inverse_asinh_transform,
    load_delta_v2_frame,
)
from pocketreg.cluster.safety import assert_compute_context, print_cluster_context  # noqa: E402
from pocketreg.models.delta_v2 import DeltaSiameseV2  # noqa: E402
from pocketreg.models.small_cnn import count_parameters  # noqa: E402
from pocketreg.paths import checkpoints_dir, plots_dir, results_dir  # noqa: E402
from pocketreg.training.metrics import mae, pearsonr, r2_score, rmse  # noqa: E402
from pocketreg.training.utils import set_seed  # noqa: E402

LOGGER = logging.getLogger("train_borzoi_delta_student_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--delta-labels", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--toy", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _limit_split(frame: pd.DataFrame, split: str, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    sub = frame[frame["split"] == split].copy()
    top_n_key = f"{split}_top_effect_n"
    max_key = f"max_{split}_rows"
    if int(cfg.get(top_n_key, 0) or 0) > 0:
        n = min(int(cfg[top_n_key]), len(sub))
        return sub.sort_values("abs_delta_teacher", ascending=False).head(n).copy()
    if float(cfg.get("top_effect_quantile_filter", 0.0) or 0.0) > 0:
        q = float(cfg["top_effect_quantile_filter"])
        cutoff = float(sub["abs_delta_teacher"].quantile(q))
        sub = sub[sub["abs_delta_teacher"] >= cutoff].copy()
    if int(cfg.get(max_key, 0) or 0) > 0 and len(sub) > int(cfg[max_key]):
        sub = sub.sample(n=int(cfg[max_key]), random_state=seed).copy()
    return sub


def _split_frames(frame: pd.DataFrame, cfg: dict[str, Any], seed: int) -> dict[str, pd.DataFrame]:
    return {split: _limit_split(frame, split, cfg, seed) for split in ("train", "val", "test")}


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, sampler_weights=None):
    import torch

    sampler = None
    if sampler_weights is not None:
        weights = torch.as_tensor(sampler_weights, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _spearman(y_true, y_pred) -> float:
    try:
        from scipy.stats import spearmanr

        value = spearmanr(y_true, y_pred).correlation
        return float(value)
    except Exception:
        return float("nan")


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return {k: float("nan") for k in ("pearson", "spearman", "r2", "mae", "rmse", "sign_accuracy")}
    nz = np.abs(y_true_arr) > 0
    sign_acc = float(np.mean(np.sign(y_true_arr[nz]) == np.sign(y_pred_arr[nz]))) if np.any(nz) else float("nan")
    return {
        "pearson": pearsonr(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "sign_accuracy": sign_acc,
    }


def top_effect_metrics(frame: pd.DataFrame, quantiles: list[float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    abs_true = frame["y_true"].abs()
    for q in quantiles:
        cutoff = float(abs_true.quantile(q))
        sub = frame[abs_true >= cutoff]
        metrics = compute_metrics(sub["y_true"].tolist(), sub["y_pred"].tolist())
        metrics["n"] = int(len(sub))
        metrics["cutoff_abs_delta"] = cutoff
        out[f"top_{int(q * 100)}"] = metrics
    return out


def effect_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = (frame["y_true"].abs().to_numpy() >= threshold).astype(int)
    score = frame["effect_score"].to_numpy()
    pred = (score >= 0.5).astype(int)
    out = {
        "threshold": float(threshold),
        "positive_fraction": float(y.mean()) if len(y) else float("nan"),
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
        out["auprc"] = float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def evaluate(model, loader, device, target_scale: float):
    import torch

    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    effect_scores: list[float] = []
    variant_ids: list[str] = []
    example_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            ref = batch["seq_ref"].to(device=device, dtype=torch.float32, non_blocking=True)
            alt = batch["seq_alt"].to(device=device, dtype=torch.float32, non_blocking=True)
            meta = batch["metadata"].to(device=device, dtype=torch.float32, non_blocking=True)
            out = model(ref, alt, meta if meta.shape[1] else None)
            pred_raw = inverse_asinh_transform(out["delta"].detach().cpu().numpy(), target_scale)
            score = torch.sigmoid(out["effect_logit"]).detach().cpu().numpy()
            y_pred.extend(float(v) for v in pred_raw)
            y_true.extend(float(v) for v in batch["delta_raw"].detach().cpu().numpy())
            effect_scores.extend(float(v) for v in score)
            variant_ids.extend(str(v) for v in batch["variant_example_id"])
            example_ids.extend(str(v) for v in batch["example_id"])
    frame = pd.DataFrame(
        {
            "variant_example_id": variant_ids,
            "example_id": example_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "effect_score": effect_scores,
        }
    )
    return compute_metrics(y_true, y_pred), frame


def save_plots(plot_dir: Path, history: list[dict[str, Any]], predictions: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    if history:
        hist = pd.DataFrame(history)
        plt.figure(figsize=(7, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        plt.plot(hist["epoch"], hist["val_pearson"], label="val_pearson")
        plt.plot(hist["epoch"], hist["val_top90_pearson"], label="val_top90_pearson")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "train_curves.png", dpi=150)
        plt.close()
    for split, frame in predictions.items():
        if frame.empty:
            continue
        plt.figure(figsize=(4, 4))
        plt.scatter(frame["y_true"], frame["y_pred"], s=3, alpha=0.2)
        plt.xlabel("teacher delta")
        plt.ylabel("student pred")
        plt.tight_layout()
        plt.savefig(plot_dir / f"delta_parity_{split}.png", dpi=150)
        plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    toy = bool(args.toy or config.get("toy", False))
    allow_local = bool(args.allow_local or train_cfg.get("allow_local", False))
    assert_compute_context("train_borzoi_delta_student_v2", allow_local=allow_local, toy=toy)
    print_cluster_context()
    set_seed(seed)

    import torch

    labels_path = args.delta_labels or Path(config.get("delta_labels_path"))
    manifest_path = args.manifest or Path(config.get("manifest_path"))
    fasta_path = args.fasta or Path(config.get("fasta_path"))
    run_name = args.run_name or config.get("run_name", "k562_delta_v2")
    run_dir = Path(config.get("output_dir", results_dir("runs"))) / run_name
    checkpoint_dir = Path(config.get("checkpoint_dir", checkpoints_dir(run_name)))
    plot_dir = Path(config.get("plot_dir", plots_dir(run_name)))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    frame = load_delta_v2_frame(labels_path, manifest_path)
    frame = frame[np.isfinite(frame["delta_teacher"])].copy()
    if frame.empty:
        raise ValueError("No finite successful delta labels")
    frames = _split_frames(frame, data_cfg, seed)
    if frames["train"].empty or frames["val"].empty:
        raise ValueError("Need non-empty train and val splits")

    train_delta = frames["train"]["delta_teacher"].to_numpy(dtype=float)
    scale_cfg = data_cfg.get("target_scale", "auto")
    target_scale = (
        estimate_delta_scale(
            train_delta,
            quantile=float(data_cfg.get("target_scale_quantile", 0.90)),
            default=float(data_cfg.get("target_scale_default", 1e-3)),
        )
        if str(scale_cfg) == "auto"
        else float(scale_cfg)
    )
    effect_threshold = float(
        np.quantile(np.abs(train_delta), float(data_cfg.get("effect_quantile", 0.90)))
    )
    sample_weights = compute_sample_weights(
        frames["train"]["abs_delta_teacher"].to_numpy(dtype=float),
        scale=target_scale,
        alpha=float(train_cfg.get("sample_weight_alpha", 4.0)),
        cap=float(train_cfg.get("sample_weight_cap", 20.0)),
    )
    frames["train"] = frames["train"].copy()
    frames["train"]["sample_weight"] = sample_weights
    for split in ("val", "test"):
        frames[split] = frames[split].copy()
        frames[split]["sample_weight"] = 1.0

    context_len = int(data_cfg.get("local_context_len", 65536))
    metadata_features = bool(data_cfg.get("metadata_features", True))
    metadata_dim = len(build_delta_metadata(frames["train"].iloc[0].to_dict())) if metadata_features else 0
    datasets = {
        split: DeltaV2Dataset(
            sub,
            fasta_path=fasta_path,
            context_len=context_len,
            target_scale=target_scale,
            effect_threshold=effect_threshold,
            metadata_features=metadata_features,
        )
        for split, sub in frames.items()
    }
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_sampler_weights = sample_weights if bool(train_cfg.get("effect_balanced_sampler", True)) else None
    loaders = {
        "train": _make_loader(datasets["train"], batch_size, num_workers, True, train_sampler_weights),
        "val": _make_loader(datasets["val"], batch_size, num_workers, False),
        "test": _make_loader(datasets["test"], batch_size, num_workers, False),
    }

    model = DeltaSiameseV2(
        metadata_dim=metadata_dim,
        channels=int(model_cfg.get("channels", 96)),
        num_blocks=int(model_cfg.get("num_blocks", 6)),
        stem_stride=int(model_cfg.get("stem_stride", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        head_hidden=int(model_cfg.get("head_hidden", 192)),
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
    max_epochs = int(train_cfg.get("max_epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))
    huber = torch.nn.HuberLoss(delta=float(train_cfg.get("huber_delta", 1.0)), reduction="none")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    effect_loss_weight = float(train_cfg.get("effect_loss_weight", 0.25))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    quantiles = [float(v) for v in config.get("eval", {}).get("top_effect_quantiles", [0.9, 0.95, 0.99])]

    (run_dir / "model_summary.txt").write_text(
        f"{model}\n\nparameters={count_parameters(model)}\ncontext_len={context_len}\n"
        f"metadata_dim={metadata_dim}\ndevice={device}\ntarget_scale={target_scale}\n"
        f"effect_threshold={effect_threshold}\nrows={{k: len(v) for k, v in frames.items()}}\n"
    )

    best_val = -np.inf
    best_epoch = -1
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            ref = batch["seq_ref"].to(device=device, dtype=torch.float32, non_blocking=True)
            alt = batch["seq_alt"].to(device=device, dtype=torch.float32, non_blocking=True)
            meta = batch["metadata"].to(device=device, dtype=torch.float32, non_blocking=True)
            y = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
            effect = batch["effect"].to(device=device, dtype=torch.float32, non_blocking=True)
            weight = batch["weight"].to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(ref, alt, meta if meta.shape[1] else None)
            loss_delta = (huber(out["delta"], y) * weight).mean()
            loss_effect = (bce(out["effect_logit"], effect) * weight).mean()
            loss = loss_delta + effect_loss_weight * loss_effect
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_metrics, val_frame = evaluate(model, loaders["val"], device, target_scale)
        val_top = top_effect_metrics(val_frame, [0.90])["top_90"]
        monitor = val_top["pearson"]
        if not np.isfinite(monitor):
            monitor = val_metrics["pearson"]
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
            "val_top90_pearson": val_top["pearson"],
            "val_top90_sign_accuracy": val_top["sign_accuracy"],
        }
        history.append(record)
        LOGGER.info(
            "epoch=%s train_loss=%.6f val_pearson=%.4f val_top90_pearson=%.4f val_top90_sign=%.4f",
            epoch,
            record["train_loss"],
            record["val_pearson"],
            record["val_top90_pearson"],
            record["val_top90_sign_accuracy"],
        )
        if np.isfinite(monitor) and monitor > best_val:
            best_val = monitor
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_scale": target_scale,
                    "effect_threshold": effect_threshold,
                    "context_len": context_len,
                    "metadata_dim": metadata_dim,
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
            "target_scale": target_scale,
            "effect_threshold": effect_threshold,
            "context_len": context_len,
            "metadata_dim": metadata_dim,
            "model_cfg": dict(model_cfg),
        },
        checkpoint_dir / "checkpoint_last.pt",
    )
    if (checkpoint_dir / "checkpoint_best.pt").exists():
        checkpoint = torch.load(checkpoint_dir / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    predictions: dict[str, pd.DataFrame] = {}
    metrics: dict[str, Any] = {
        "run_name": run_name,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_top90_pearson": best_val,
        "runtime_seconds": time.time() - started,
        "target_scale": target_scale,
        "effect_threshold": effect_threshold,
        "checkpoint_dir": str(checkpoint_dir),
        "plot_dir": str(plot_dir),
        "params_train": count_parameters(model),
        "num_rows": {split: int(len(sub)) for split, sub in frames.items()},
    }
    for split, loader in loaders.items():
        split_metrics, frame_pred = evaluate(model, loader, device, target_scale)
        metrics[split] = split_metrics
        metrics[f"{split}_top_effect"] = top_effect_metrics(frame_pred, quantiles)
        metrics[f"{split}_effect_classification"] = effect_metrics(frame_pred, effect_threshold)
        predictions[split] = frame_pred
        frame_pred.to_parquet(run_dir / f"predictions_{split}.parquet", index=False)
    pd.DataFrame(history).to_csv(run_dir / "train_log.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    save_plots(plot_dir, history, predictions)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
