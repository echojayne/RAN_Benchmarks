"""Run a paper-aligned Milan cell-level forecasting benchmark.

This script implements a tractable benchmark aligned to:

- single-cell prediction
- one-step-ahead forecasting
- per-cell min-max normalized traffic load
- time features: day-of-week and hour-of-day
- training windows matching the paper's first-week and five-week setups

The implementation intentionally stays separate from the main traffic-forecasting package
`24 -> 6` multivariate benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from traffic_prediction_core.data.prepare_milan import (
    _aggregate_hourly_rows,
    _list_raw_sources,
    _read_source_frame,
    load_yaml,
)
from traffic_prediction_core.train.common import resolve_device, set_seed


def _timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


class CellWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32, copy=False))
        self.targets = torch.from_numpy(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.inputs[index], "targets": self.targets[index]}


class CellNaiveRegressor(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1, 0]


class CellLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(inputs)
        return self.head(hidden[-1]).squeeze(-1)


class CellITransformerRegressor(nn.Module):
    def __init__(self, context_len: int, num_tokens: int, d_model: int, depth: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_proj = nn.Linear(context_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.num_tokens = num_tokens

    def forward(self, inputs: torch.Tensor, *, active_layers: int | None = None) -> torch.Tensor:
        total_layers = len(self.encoder.layers)
        if active_layers is None:
            active_layers = total_layers
        if active_layers < 1 or active_layers > total_layers:
            raise ValueError(f"active_layers must be in [1, {total_layers}], got {active_layers}")
        tokens = inputs.transpose(1, 2).contiguous()
        tokens = self.token_proj(tokens)
        encoded = tokens
        for layer in list(self.encoder.layers)[:active_layers]:
            encoded = layer(encoded)
        # Use the traffic token representation after attending to time-feature tokens.
        return self.output_head(encoded[:, 0, :]).squeeze(-1)


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-steps", type=int, default=30)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    return parser.parse_args()


def _select_cells(pivot: pd.DataFrame, config: dict[str, Any]) -> list[int]:
    bench_cfg = config["benchmark"]
    top_k = int(bench_cfg.get("top_k_cells", 32))
    if args_cells := int(bench_cfg.get("cells_override", 0)):
        top_k = args_cells
    include_cells = [int(value) for value in bench_cfg.get("include_cells", [])]
    active_scores = pivot.sum(axis=0).sort_values(ascending=False)
    selected = [int(value) for value in active_scores.head(top_k).index.tolist()]
    union = sorted(set(selected) | set(include_cells))
    filtered: list[int] = []
    for cell_id in union:
        if cell_id not in pivot.columns:
            continue
        series = pivot[cell_id].to_numpy(dtype=np.float32)
        if float(np.nanmax(series) - np.nanmin(series)) > 1e-6:
            filtered.append(cell_id)
    return filtered


def _load_candidate_cells(config: dict[str, Any]) -> list[int]:
    bench_cfg = config["benchmark"]
    candidate_path = bench_cfg.get("candidate_cells_path")
    if not candidate_path:
        return []
    path = Path(os.path.expandvars(str(candidate_path))).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"candidate_cells_path does not exist: {path}")
    frame = pd.read_csv(path)
    if "region_id" not in frame.columns:
        raise ValueError(f"candidate_cells_path must include region_id column: {path}")
    return [int(value) for value in frame["region_id"].tolist()]


def _build_filtered_hourly_pivot(
    dataset_cfg: dict[str, Any],
    *,
    candidate_cells: list[int],
    cache_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        index_ns = cached["index_ns"].astype(np.int64)
        columns = cached["columns"].astype(np.int64)
        values = cached["values"].astype(np.float32)
        pivot = pd.DataFrame(values, index=pd.to_datetime(index_ns, utc=True), columns=columns)
        raw_sources = [str(value) for value in cached["raw_sources"]]
        return pivot, raw_sources

    selected_set = set(candidate_cells)
    target_col = str(dataset_cfg.get("target_field", "internet_traffic"))
    region_col = str(dataset_cfg.get("region_column", "square_id"))
    aggregate_minutes = int(dataset_cfg.get("aggregate_minutes", 60))
    candidates = _list_raw_sources(dataset_cfg)
    hourly_parts: list[pd.DataFrame] = []
    raw_sources: list[str] = []
    for index, path in enumerate(candidates, start=1):
        frame = _read_source_frame(path, dataset_cfg)
        frame = frame[frame[region_col].isin(selected_set)].copy()
        if frame.empty:
            continue
        hourly = _aggregate_hourly_rows(frame, dataset_cfg)
        hourly_parts.append(hourly)
        raw_sources.append(str(path))
        print(json.dumps({"stage": "aggregate_source", "index": index, "total": len(candidates), "source": path.name, "selected_rows": int(len(frame))}), flush=True)
    if not hourly_parts:
        raise ValueError("filtered hourly aggregation produced no rows")
    hourly = (
        pd.concat(hourly_parts, ignore_index=True)
        .groupby(["bucket", region_col], as_index=False)[target_col]
        .sum()
        .sort_values(["bucket", region_col])
    )
    pivot = hourly.pivot(index="bucket", columns=region_col, values=target_col).sort_index()
    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq=f"{aggregate_minutes}min", tz="UTC")
    pivot = pivot.reindex(full_index, fill_value=0.0)
    pivot.columns = pivot.columns.astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        index_ns=pivot.index.view("int64"),
        columns=pivot.columns.to_numpy(dtype=np.int64),
        values=pivot.to_numpy(dtype=np.float32),
        raw_sources=np.asarray(raw_sources, dtype=str),
    )
    return pivot, raw_sources


def _build_regimes(config: dict[str, Any]) -> list[RegimeSpec]:
    bench_cfg = config["benchmark"]
    return [
        RegimeSpec(
            name="first_week",
            train_start=_timestamp(str(bench_cfg["fine_tune_start"])),
            train_end=_timestamp(str(bench_cfg["first_week_end"])),
            test_start=_timestamp(str(bench_cfg["test_start"])),
            test_end=_timestamp(str(bench_cfg["test_end"])),
        ),
        RegimeSpec(
            name="five_weeks",
            train_start=_timestamp(str(bench_cfg["fine_tune_start"])),
            train_end=_timestamp(str(bench_cfg["five_weeks_end"])),
            test_start=_timestamp(str(bench_cfg["test_start"])),
            test_end=_timestamp(str(bench_cfg["test_end"])),
        ),
    ]


def _split_train_val(inputs: np.ndarray, targets: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(inputs.shape[0])
    if total < 20:
        raise ValueError(f"too few training samples: {total}")
    val_count = max(16, int(math.ceil(total * val_fraction)))
    val_count = min(val_count, max(1, total // 5))
    train_count = total - val_count
    return inputs[:train_count], targets[:train_count], inputs[train_count:], targets[train_count:]


def _make_loaders(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CellWindowDataset(train_inputs, train_targets)
    val_dataset = CellWindowDataset(val_inputs, val_targets)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()),
    )


def _run_epoch(model: nn.Module, loader: DataLoader, *, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            pred = model(inputs)
            loss = torch.mean((pred - targets) ** 2)
            mae = torch.mean(torch.abs(pred - targets))
            mse = torch.mean((pred - targets) ** 2)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_mae += float(mae.item()) * batch_size
        total_mse += float(mse.item()) * batch_size
        total_count += batch_size
    mean_mse = total_mse / max(1, total_count)
    return {
        "loss": total_loss / max(1, total_count),
        "MAE": total_mae / max(1, total_count),
        "MSE": mean_mse,
        "RMSE": float(math.sqrt(mean_mse)),
    }


def _fit_model(
    model_name: str,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    model_cfg = cfg["models"][model_name]
    if model_name == "lstm":
        model = CellLSTMRegressor(
            input_size=int(train_inputs.shape[2]),
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    elif model_name == "itransformer":
        model = CellITransformerRegressor(
            context_len=int(train_inputs.shape[1]),
            num_tokens=int(train_inputs.shape[2]),
            d_model=int(model_cfg.get("d_model", 96)),
            depth=int(model_cfg.get("depth", 3)),
            num_heads=int(model_cfg.get("num_heads", 3)),
            ffn_dim=int(model_cfg.get("ffn_dim", 192)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"unsupported model: {model_name}")

    model = model.to(device)
    training_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", training_cfg.get("learning_rate", 1e-3))),
        weight_decay=float(model_cfg.get("weight_decay", training_cfg.get("weight_decay", 1e-4))),
    )
    train_loader, val_loader = _make_loaders(
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        batch_size=int(training_cfg.get("batch_size", 64)),
    )

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    patience = int(training_cfg.get("patience", 5))
    epochs = int(training_cfg.get("epochs", 20))
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_val_mae": best_val_mae, "history": history, "seed": seed}


def _evaluate_naive(test_inputs: np.ndarray, test_targets: np.ndarray) -> dict[str, Any]:
    pred = test_inputs[:, -1, 0]
    target = test_targets
    diff = pred - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    return {
        "MAE": mae,
        "RMSE": float(math.sqrt(mse)),
        "MSE": mse,
        "predictions": pred,
        "targets": target,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _benchmark_latency(
    model: nn.Module,
    sample_inputs: np.ndarray,
    *,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    inputs = torch.from_numpy(sample_inputs[:batch_size].astype(np.float32, copy=False)).to(device=device)
    model.eval()
    for _ in range(warmup_steps):
        model(inputs)
    _synchronize(device)
    timings_ms = []
    for _ in range(measure_steps):
        start = time.perf_counter()
        model(inputs)
        _synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_latency_ms": float(np.median(timings_ms)),
        "mean_latency_ms": float(np.mean(timings_ms)),
        "p90_latency_ms": float(np.percentile(timings_ms, 90.0)),
        "batch_size": int(inputs.shape[0]),
    }


@torch.no_grad()
def _evaluate_model(model: nn.Module, test_inputs: np.ndarray, test_targets: np.ndarray, *, device: torch.device, batch_size: int) -> dict[str, Any]:
    loader = DataLoader(
        CellWindowDataset(test_inputs, test_targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        pred = model(inputs)
        preds.append(pred.detach().cpu().numpy())
        tgts.append(targets.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(tgts, axis=0)
    diff = pred - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    return {
        "MAE": mae,
        "RMSE": float(math.sqrt(mse)),
        "MSE": mse,
        "predictions": pred,
        "targets": target,
    }


def _generate_windows(
    cell_series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    context_len: int,
    regime: RegimeSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loads = cell_series.astype(np.float32, copy=False)
    dow = (timestamps.dayofweek.to_numpy(dtype=np.float32) / 6.0).astype(np.float32, copy=False)
    hod = (timestamps.hour.to_numpy(dtype=np.float32) / 23.0).astype(np.float32, copy=False)
    features = np.stack([loads, dow, hod], axis=1)

    train_inputs: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    test_inputs: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    for end in range(context_len - 1, len(timestamps) - 1):
        target_index = end + 1
        target_time = timestamps[target_index]
        sample = features[end - context_len + 1 : end + 1]
        target = features[target_index, 0]
        if regime.train_start <= target_time < regime.train_end:
            train_inputs.append(sample)
            train_targets.append(target)
        elif regime.test_start <= target_time < regime.test_end:
            test_inputs.append(sample)
            test_targets.append(target)
    if not train_inputs or not test_inputs:
        raise ValueError(f"empty sample set for regime {regime.name}")
    return (
        np.asarray(train_inputs, dtype=np.float32),
        np.asarray(train_targets, dtype=np.float32),
        np.asarray(test_inputs, dtype=np.float32),
        np.asarray(test_targets, dtype=np.float32),
    )


def _normalize_cell(series: pd.Series, *, start_time: pd.Timestamp, end_time: pd.Timestamp) -> np.ndarray | None:
    focus = series.loc[(series.index >= start_time) & (series.index < end_time)].astype(np.float32)
    if focus.empty:
        return None
    min_value = float(focus.min())
    max_value = float(focus.max())
    scale = max_value - min_value
    if scale <= 1e-6:
        return None
    normalized = (series.astype(np.float32) - min_value) / scale
    return normalized.to_numpy(dtype=np.float32, copy=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs > 0:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.cells > 0:
        config.setdefault("benchmark", {})["cells_override"] = args.cells
    if args.seed > 0:
        config.setdefault("training", {})["seed"] = args.seed

    output_dir = Path(os.path.expandvars(str(config.get("output_dir", "outputs/traffic_prediction/paper_aligned"))))
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = config["training"]
    seed = int(training_cfg.get("seed", 17))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = resolve_device(str(training_cfg.get("device", "auto")))

    source_prepare_config = Path(os.path.expandvars(str(config["source_prepare_config"]))).expanduser()
    base_prepare_cfg = load_yaml(source_prepare_config)
    include_cells = [int(value) for value in config["benchmark"].get("include_cells", [])]
    candidate_cells = sorted(set(_load_candidate_cells(config)) | set(include_cells))
    if not candidate_cells:
        raise ValueError("paper-aligned benchmark requires benchmark.candidate_cells_path to avoid full 10k-cell re-aggregation")
    pivot, raw_sources = _build_filtered_hourly_pivot(
        base_prepare_cfg["dataset"],
        candidate_cells=candidate_cells,
        cache_path=output_dir / "filtered_hourly_cache.npz",
    )
    bench_cfg = config["benchmark"]
    start_time = _timestamp(str(bench_cfg["series_start"]))
    series_end = _timestamp(str(bench_cfg["series_end"]))
    pivot = pivot.loc[(pivot.index >= start_time) & (pivot.index < series_end)].copy()
    selected_cells = _select_cells(pivot, config)
    regimes = _build_regimes(config)

    per_cell_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []

    for regime in regimes:
        for model_name in ["naive", "lstm", "itransformer"]:
            model_maes: list[float] = []
            model_rmses: list[float] = []
            model_mses: list[float] = []
            latency_model: nn.Module | None = None
            latency_reference_inputs: np.ndarray | None = None
            for cell_id in selected_cells:
                normalized = _normalize_cell(
                    pivot[cell_id],
                    start_time=_timestamp(str(bench_cfg["normalization_start"])),
                    end_time=_timestamp(str(bench_cfg["normalization_end"])),
                )
                if normalized is None:
                    continue
                train_inputs, train_targets, test_inputs, test_targets = _generate_windows(
                    normalized,
                    pivot.index,
                    context_len=int(bench_cfg.get("context_len", 3)),
                    regime=regime,
                )
                train_inputs, train_targets, val_inputs, val_targets = _split_train_val(
                    train_inputs,
                    train_targets,
                    val_fraction=float(training_cfg.get("val_fraction", 0.1)),
                )

                if model_name == "naive":
                    eval_result = _evaluate_naive(test_inputs, test_targets)
                    train_meta: dict[str, Any] = {"best_val_mae": float(np.mean(np.abs(val_inputs[:, -1, 0] - val_targets)))}
                    if latency_model is None:
                        latency_model = CellNaiveRegressor().to(device)
                        latency_reference_inputs = test_inputs
                else:
                    model, train_meta = _fit_model(
                        model_name,
                        train_inputs,
                        train_targets,
                        val_inputs,
                        val_targets,
                        config,
                        device=device,
                        seed=seed,
                    )
                    eval_result = _evaluate_model(
                        model,
                        test_inputs,
                        test_targets,
                        device=device,
                        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 64))),
                    )
                    if latency_model is None:
                        latency_model = model
                        latency_reference_inputs = test_inputs

                per_cell_rows.append(
                    {
                        "regime": regime.name,
                        "model": model_name,
                        "cell_id": int(cell_id),
                        "train_samples": int(train_inputs.shape[0]),
                        "val_samples": int(val_inputs.shape[0]),
                        "test_samples": int(test_inputs.shape[0]),
                        "best_val_mae": float(train_meta["best_val_mae"]),
                        "test_MAE": float(eval_result["MAE"]),
                        "test_RMSE": float(eval_result["RMSE"]),
                        "test_MSE": float(eval_result["MSE"]),
                    }
                )
                model_maes.append(float(eval_result["MAE"]))
                model_rmses.append(float(eval_result["RMSE"]))
                model_mses.append(float(eval_result["MSE"]))

            if not model_maes:
                continue
            aggregate_rows.append(
                {
                    "regime": regime.name,
                    "model": model_name,
                    "num_cells": len(model_maes),
                    "mean_MAE": float(np.mean(model_maes)),
                    "median_MAE": float(np.median(model_maes)),
                    "p90_MAE": float(np.percentile(model_maes, 90)),
                    "mean_RMSE": float(np.mean(model_rmses)),
                    "mean_MSE": float(np.mean(model_mses)),
                }
            )
            if latency_model is not None and latency_reference_inputs is not None:
                latency_rows.append(
                    {
                        "regime": regime.name,
                        "model": model_name,
                        **_benchmark_latency(
                            latency_model,
                            latency_reference_inputs,
                            device=device,
                            warmup_steps=int(args.latency_warmup),
                            measure_steps=int(args.latency_steps),
                            batch_size=int(args.latency_batch_size),
                        ),
                    }
                )

    _write_csv(output_dir / "per_cell_metrics.csv", per_cell_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(output_dir / "latency_metrics.csv", latency_rows)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "source_prepare_config": config["source_prepare_config"],
                "selected_cells": selected_cells,
                "num_selected_cells": len(selected_cells),
                "raw_sources": raw_sources,
                "aggregate_metrics": aggregate_rows,
                "latency_metrics": latency_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "num_cells": len(selected_cells), "aggregate_metrics": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
