"""Shared evaluation helpers for original traffic-forecasting baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_prediction_core.data import build_split_dataloader, load_manifest
from traffic_prediction_core.train.common import load_yaml


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def build_eval_loader(
    config: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    limit: int,
) -> Any:
    return build_split_dataloader(
        config,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=limit,
        normalize=True,
    )


def load_normalization_stats(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    manifest = load_manifest(config)
    mean = np.asarray(manifest["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["normalization"]["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def denormalize_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_arr = mean.reshape(1, 1, -1)
    std_arr = std.reshape(1, 1, -1)
    prediction_np = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float32, copy=False)
    return prediction_np * std_arr + mean_arr, target_np * std_arr + mean_arr


def summarize_regression(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    error = predictions - targets
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "RMSE": rmse, "MSE": mse}


def per_sample_metrics(predictions: np.ndarray, targets: np.ndarray) -> list[dict[str, float]]:
    error = predictions - targets
    mae = np.mean(np.abs(error), axis=(1, 2))
    mse = np.mean(error**2, axis=(1, 2))
    rmse = np.sqrt(mse)
    rows: list[dict[str, float]] = []
    for sample_mae, sample_rmse, sample_mse in zip(mae, rmse, mse, strict=True):
        rows.append(
            {
                "MAE": float(sample_mae),
                "RMSE": float(sample_rmse),
                "MSE": float(sample_mse),
            }
        )
    return rows


def write_eval_outputs(
    output_dir: str | Path,
    *,
    model_name: str,
    split: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {"model": model_name, "split": split, **summarize_regression(predictions, targets)}

    predictions_path = output_path / "predictions.npz"
    np.savez_compressed(predictions_path, predictions=predictions, targets=targets)

    summary_json = output_path / "metrics_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = output_path / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "MAE", "RMSE", "MSE"])
        writer.writeheader()
        writer.writerow(summary)

    sample_rows = per_sample_metrics(predictions, targets)
    per_sample_path = output_path / "per_sample_metrics.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_index", "target_start_time_ns", "MAE", "RMSE", "MSE"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for meta, metrics in zip(metadata, sample_rows, strict=True):
            writer.writerow(
                {
                    "sample_index": int(meta["sample_index"]),
                    "target_start_time_ns": int(meta["target_start_time_ns"]),
                    **metrics,
                }
            )

    run_summary = output_path / "run_summary.json"
    run_summary.write_text(
        json.dumps({"summary": summary, "num_samples": int(predictions.shape[0])}, indent=2),
        encoding="utf-8",
    )

    return {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "per_sample_csv": per_sample_path,
        "predictions": predictions_path,
        "run_summary": run_summary,
    }
