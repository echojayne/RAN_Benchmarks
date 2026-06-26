from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


RAW_COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
]


@dataclass
class CitywideITransformerConfig:
    raw_data_dir: str
    output_dir: str
    cache_dir: str
    exclude_cell_ids_path: str | None = None
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    benchmark_start: str | None = None
    train_weeks: int | None = None
    test_weeks: int | None = None
    train_ratio: float = 0.7
    val_ratio_within_train: float = 0.02
    context_len: int = 3
    horizon: int = 1
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    depth: int = 4
    num_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1
    cell_embedding_dim: int = 16
    cell_batch_size: int = 10000
    eval_cell_batch_size: int = 10000
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _cache_paths(cache_dir: Path) -> Tuple[Path, Path]:
    return (
        cache_dir / "hourly_internet_matrix.npy",
        cache_dir / "hourly_internet_meta.json",
    )


def prepare_hourly_matrix(raw_data_dir: Path, cache_dir: Path) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    ensure_dir(cache_dir)
    matrix_path, meta_path = _cache_paths(cache_dir)
    if matrix_path.exists() and meta_path.exists():
        matrix = np.load(matrix_path)
        meta = json.loads(meta_path.read_text())
        timestamps = pd.to_datetime(meta["timestamps"], utc=True)
        cell_ids = np.asarray(meta["cell_ids"], dtype=np.int64)
        return matrix.astype(np.float32), timestamps, cell_ids

    raw_files = sorted(raw_data_dir.glob("sms-call-internet-mi-*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"no raw Milan files found under {raw_data_dir}")

    frames: List[pd.DataFrame] = []
    for raw_file in raw_files:
        df = pd.read_csv(
            raw_file,
            sep="\t",
            header=None,
            names=RAW_COLUMNS,
            usecols=[0, 1, 7],
        )
        df["timestamp"] = pd.to_datetime(df["time_interval"], unit="ms", utc=True).dt.floor("h")
        agg = (
            df.groupby(["timestamp", "square_id"], sort=True)["internet_traffic"]
            .sum()
            .reset_index()
        )
        frames.append(agg)

    full_df = pd.concat(frames, ignore_index=True)
    pivot = full_df.pivot_table(
        index="timestamp",
        columns="square_id",
        values="internet_traffic",
        aggfunc="sum",
        fill_value=0.0,
    )
    all_timestamps = pd.date_range(
        start=pivot.index.min(),
        end=pivot.index.max(),
        freq="h",
        tz="UTC",
    )
    all_cells = np.arange(1, 10001, dtype=np.int64)
    pivot = pivot.reindex(index=all_timestamps, columns=all_cells, fill_value=0.0)
    matrix = pivot.to_numpy(dtype=np.float32)

    np.save(matrix_path, matrix)
    meta_path.write_text(
        json.dumps(
            {
                "timestamps": [ts.isoformat() for ts in all_timestamps],
                "cell_ids": all_cells.tolist(),
                "shape": list(matrix.shape),
            },
            indent=2,
        )
    )
    return matrix, all_timestamps, all_cells


def load_excluded_cell_ids(path: str | None) -> np.ndarray:
    if not path:
        return np.asarray([], dtype=np.int64)
    payload = json.loads(Path(path).read_text())
    return np.asarray(payload.get("cell_ids", []), dtype=np.int64)


def build_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hours = timestamps.hour.to_numpy()
    dows = timestamps.dayofweek.to_numpy()
    hour_angle = 2.0 * math.pi * hours / 24.0
    dow_angle = 2.0 * math.pi * dows / 7.0
    features = np.stack(
        [
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def compute_per_cell_scaler(train_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = train_matrix.min(axis=0)
    maxs = train_matrix.max(axis=0)
    scales = np.maximum(maxs - mins, 1e-6)
    return mins.astype(np.float32), scales.astype(np.float32)


def _to_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _find_timestamp_index(timestamps: pd.DatetimeIndex, value: str) -> int:
    target = _to_utc_timestamp(value)
    matches = np.where(timestamps == target)[0]
    if matches.size == 0:
        raise ValueError(f"timestamp {target.isoformat()} not found in hourly Milan index")
    return int(matches[0])


def resolve_time_splits(config: CitywideITransformerConfig, timestamps: pd.DatetimeIndex) -> Dict[str, int]:
    total_steps = len(timestamps)
    if config.train_start and config.train_end and config.test_start and config.test_end:
        train_start_idx = _find_timestamp_index(timestamps, config.train_start)
        train_end_idx = _find_timestamp_index(timestamps, config.train_end)
        test_start_idx = _find_timestamp_index(timestamps, config.test_start)
        test_end_idx = _find_timestamp_index(timestamps, config.test_end)
    elif config.benchmark_start and config.train_weeks and config.test_weeks:
        train_start_idx = _find_timestamp_index(timestamps, config.benchmark_start)
        train_hours = int(config.train_weeks) * 7 * 24
        test_hours = int(config.test_weeks) * 7 * 24
        train_end_idx = train_start_idx + train_hours
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_hours
        if test_end_idx > total_steps:
            raise ValueError(
                f"requested benchmark window ends at index {test_end_idx}, "
                f"but only {total_steps} hourly steps are available"
            )
    else:
        train_start_idx = 0
        train_end_idx = int(total_steps * config.train_ratio)
        test_start_idx = train_end_idx
        test_end_idx = total_steps

    val_steps = int((train_end_idx - train_start_idx) * config.val_ratio_within_train)
    val_steps = max(val_steps, 1)
    val_start_idx = train_end_idx - val_steps

    min_train_end = train_start_idx + config.context_len + 1
    if val_start_idx < min_train_end:
        raise ValueError("validation split leaves too few train targets for the chosen context length")
    if test_start_idx < train_start_idx + config.context_len:
        raise ValueError("test split starts before enough context is available")
    if test_end_idx <= test_start_idx:
        raise ValueError("test split is empty")
    if train_end_idx != test_start_idx:
        raise ValueError("expected contiguous train/test boundary for this benchmark")

    return {
        "train_start_idx": train_start_idx,
        "val_start_idx": val_start_idx,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "test_end_idx": test_end_idx,
    }


class InvertedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout(attn_out)
        ffn_in = self.norm2(x)
        x = x + self.dropout(self.ffn(ffn_in))
        return x


class CitywideITransformer(nn.Module):
    def __init__(
        self,
        context_len: int,
        num_variates: int,
        num_cells: int,
        d_model: int,
        depth: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        cell_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(context_len, d_model)
        self.var_embedding = nn.Parameter(torch.zeros(num_variates, d_model))
        self.blocks = nn.ModuleList(
            [
                InvertedTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.cell_embedding = nn.Embedding(num_cells, cell_embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model + cell_embedding_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, cell_ids: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.var_embedding.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        cell_embed = self.cell_embedding(cell_ids)
        x = torch.cat([x, cell_embed], dim=-1)
        return self.head(x).squeeze(-1)


def make_batch(
    norm_matrix: np.ndarray,
    time_features: np.ndarray,
    target_time: int,
    cell_indices: np.ndarray,
    context_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    traffic_hist = norm_matrix[target_time - context_len : target_time, cell_indices].T[:, :, None]
    shared_time = np.broadcast_to(
        time_features[target_time - context_len : target_time][None, :, :],
        (cell_indices.shape[0], context_len, time_features.shape[-1]),
    )
    x = np.concatenate([traffic_hist, shared_time], axis=-1)
    y = norm_matrix[target_time, cell_indices]
    return x.astype(np.float32), y.astype(np.float32)


def evaluate_on_times(
    model: nn.Module,
    norm_matrix: np.ndarray,
    time_features: np.ndarray,
    target_times: np.ndarray,
    context_len: int,
    eval_cell_batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    maes: List[float] = []
    with torch.no_grad():
        all_cells = np.arange(norm_matrix.shape[1], dtype=np.int64)
        for target_time in target_times:
            for start in range(0, all_cells.shape[0], eval_cell_batch_size):
                batch_cells = all_cells[start : start + eval_cell_batch_size]
                x_np, y_np = make_batch(norm_matrix, time_features, int(target_time), batch_cells, context_len)
                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(y_np).to(device)
                cell_ids = torch.from_numpy(batch_cells).to(device)
                pred = model(x, cell_ids)
                loss = torch.mean((pred - y) ** 2)
                mae = torch.mean(torch.abs(pred - y))
                losses.append(float(loss.detach().cpu()))
                maes.append(float(mae.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "mae": float(np.mean(maes)),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0.0 else 0.0
    y_range = float(np.max(y_true) - np.min(y_true))
    nmae = float(mae / y_range) if y_range > 0.0 else 0.0
    nrmse = float(rmse / y_range) if y_range > 0.0 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nmae": nmae,
        "nrmse": nrmse,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def append_train_log(path: Path, row: Dict[str, float]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_citywide_itransformer(config: CitywideITransformerConfig, resume: bool = False) -> Dict[str, float]:
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    cache_dir = Path(config.cache_dir)
    checkpoint_dir = output_dir / "checkpoints"
    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    ensure_dir(checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    matrix, timestamps, cell_ids = prepare_hourly_matrix(Path(config.raw_data_dir), cache_dir)
    excluded_cell_ids = load_excluded_cell_ids(config.exclude_cell_ids_path)
    if excluded_cell_ids.size > 0:
        keep_mask = ~np.isin(cell_ids, excluded_cell_ids)
        matrix = matrix[:, keep_mask]
        cell_ids = cell_ids[keep_mask]
    total_steps, num_cells = matrix.shape
    split = resolve_time_splits(config, timestamps)
    train_start_idx = split["train_start_idx"]
    val_start_idx = split["val_start_idx"]
    train_end_idx = split["train_end_idx"]
    test_start_idx = split["test_start_idx"]
    test_end_idx = split["test_end_idx"]

    mins, scales = compute_per_cell_scaler(matrix[train_start_idx:val_start_idx])
    norm_matrix = ((matrix - mins[None, :]) / scales[None, :]).astype(np.float32)
    time_features = build_time_features(timestamps)

    train_target_times = np.arange(train_start_idx + config.context_len, val_start_idx, dtype=np.int64)
    val_target_times = np.arange(val_start_idx, train_end_idx, dtype=np.int64)
    test_target_times = np.arange(test_start_idx, test_end_idx, dtype=np.int64)

    model = CitywideITransformer(
        context_len=config.context_len,
        num_variates=1 + time_features.shape[-1],
        num_cells=num_cells,
        d_model=config.d_model,
        depth=config.depth,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        cell_embedding_dim=config.cell_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_mae = float("inf")
    best_epoch = 0
    start_epoch = 1
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    train_log_path = output_dir / "train_log.csv"

    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val_mae = float(state["best_val_mae"])
        best_epoch = int(state["best_epoch"])
        print(f"resuming_from_epoch={start_epoch}", flush=True)

    save_json(
        output_dir / "protocol_manifest.json",
        {
            "config": asdict(config),
            "total_steps": total_steps,
            "num_cells": num_cells,
            "num_excluded_cells": int(excluded_cell_ids.size),
            "excluded_cell_ids_path": config.exclude_cell_ids_path,
            "train_start_idx": train_start_idx,
            "val_start_idx": val_start_idx,
            "train_end_idx": train_end_idx,
            "test_start_idx": test_start_idx,
            "test_end_idx": test_end_idx,
            "train_target_steps": int(train_target_times.shape[0]),
            "val_target_steps": int(val_target_times.shape[0]),
            "test_target_steps": int(test_target_times.shape[0]),
            "timestamps": {
                "dataset_start": timestamps[0].isoformat(),
                "train_start": timestamps[train_start_idx].isoformat(),
                "val_start": timestamps[val_start_idx].isoformat(),
                "train_end_exclusive": timestamps[train_end_idx].isoformat(),
                "test_start": timestamps[test_start_idx].isoformat(),
                "test_end_exclusive": timestamps[test_end_idx].isoformat()
                if test_end_idx < total_steps
                else (timestamps[-1] + pd.Timedelta(hours=1)).isoformat(),
                "dataset_end": timestamps[-1].isoformat(),
            },
        },
    )

    loss_fn = nn.MSELoss()
    all_cells = np.arange(num_cells, dtype=np.int64)
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_maes: List[float] = []
        shuffled_times = np.random.permutation(train_target_times)
        for step, target_time in enumerate(shuffled_times, start=1):
            if config.cell_batch_size >= num_cells:
                batch_cells = all_cells
            else:
                batch_cells = np.random.choice(all_cells, size=config.cell_batch_size, replace=False)
            x_np, y_np = make_batch(norm_matrix, time_features, int(target_time), batch_cells, config.context_len)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            cell_tensor = torch.from_numpy(batch_cells).to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, cell_tensor)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            mae = torch.mean(torch.abs(pred - y))
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_maes.append(float(mae.detach().cpu()))

            if step % 100 == 0 or step == len(shuffled_times):
                print(
                    f"epoch={epoch} step={step}/{len(shuffled_times)} loss={epoch_losses[-1]:.6f} mae={epoch_maes[-1]:.6f}",
                    flush=True,
                )

        val_metrics = evaluate_on_times(
            model=model,
            norm_matrix=norm_matrix,
            time_features=time_features,
            target_times=val_target_times,
            context_len=config.context_len,
            eval_cell_batch_size=config.eval_cell_batch_size,
            device=device,
        )
        train_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_mae": float(np.mean(epoch_maes)),
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        append_train_log(train_log_path, train_row)
        print(
            f"epoch={epoch} train_loss={train_row['train_loss']:.6f} train_mae={train_row['train_mae']:.6f} "
            f"val_loss={train_row['val_loss']:.6f} val_mae={train_row['val_mae']:.6f}",
            flush=True,
        )

        if train_row["val_mae"] < best_val_mae:
            best_val_mae = train_row["val_mae"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                best_path,
            )

        torch.save(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_mae": best_val_mae,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            last_path,
        )

    if best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model"])
        best_epoch = int(best_state["best_epoch"])
        best_val_mae = float(best_state["best_val_mae"])

    num_test_steps = test_target_times.shape[0]
    preds_norm = np.zeros((num_test_steps, num_cells), dtype=np.float32)
    trues_norm = np.zeros((num_test_steps, num_cells), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for row_idx, target_time in enumerate(test_target_times):
            for start in range(0, num_cells, config.eval_cell_batch_size):
                batch_cells = all_cells[start : start + config.eval_cell_batch_size]
                x_np, y_np = make_batch(norm_matrix, time_features, int(target_time), batch_cells, config.context_len)
                x = torch.from_numpy(x_np).to(device)
                cell_tensor = torch.from_numpy(batch_cells).to(device)
                pred = model(x, cell_tensor).detach().cpu().numpy().astype(np.float32)
                preds_norm[row_idx, batch_cells] = pred
                trues_norm[row_idx, batch_cells] = y_np

            if (row_idx + 1) % 50 == 0 or (row_idx + 1) == num_test_steps:
                print(f"eval_step={row_idx + 1}/{num_test_steps}", flush=True)

    preds_raw = preds_norm * scales[None, :] + mins[None, :]
    trues_raw = trues_norm * scales[None, :] + mins[None, :]

    per_cell_rows: List[Dict[str, float]] = []
    for cell_idx, cell_id in enumerate(cell_ids):
        raw_metrics = compute_regression_metrics(trues_raw[:, cell_idx], preds_raw[:, cell_idx])
        norm_metrics = compute_regression_metrics(trues_norm[:, cell_idx], preds_norm[:, cell_idx])
        per_cell_rows.append(
            {
                "cell_id": int(cell_id),
                "mae": raw_metrics["mae"],
                "rmse": raw_metrics["rmse"],
                "r2": raw_metrics["r2"],
                "nmae": norm_metrics["mae"],
                "nrmse": norm_metrics["rmse"],
            }
        )

    per_cell_path = output_dir / "per_cell_metrics.csv"
    with per_cell_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_cell_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_cell_rows)

    metrics_df = pd.DataFrame(per_cell_rows)
    summary = {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "mean_mae": float(metrics_df["mae"].mean()),
        "median_mae": float(metrics_df["mae"].median()),
        "mean_rmse": float(metrics_df["rmse"].mean()),
        "median_rmse": float(metrics_df["rmse"].median()),
        "mean_r2": float(metrics_df["r2"].mean()),
        "median_r2": float(metrics_df["r2"].median()),
        "mean_nmae": float(metrics_df["nmae"].mean()),
        "median_nmae": float(metrics_df["nmae"].median()),
        "mean_nrmse": float(metrics_df["nrmse"].mean()),
        "median_nrmse": float(metrics_df["nrmse"].median()),
    }
    save_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
