"""Published ML-TP style baseline utilities for the paper-aligned Milan benchmark."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_prediction_core.paper.run_cell_benchmark import CellWindowDataset, _evaluate_model, _make_loaders, _run_epoch


class PaperMLTPLSTMRegressor(nn.Module):
    """Three-layer cascade LSTM base-learner from the ML-TP paper."""

    def __init__(self, input_size: int = 3, hidden_sizes: tuple[int, int, int] = (5, 5, 1)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(
                nn.LSTM(
                    input_size=current_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            )
            current_size = hidden_size
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x, _ = layer(x)
        return x[:, -1, -1]


@dataclass(frozen=True)
class MLTPSample:
    cell_id: int
    meta_feature: np.ndarray
    state_dict: dict[str, torch.Tensor]


def build_weekly_meta_feature(series: np.ndarray, *, period: int = 168, frequency_bins: tuple[int, ...] = (1, 7, 14, 21, 28)) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.size == 0:
        raise ValueError("cannot build meta-feature from empty series")
    if values.size < period:
        repeats = int(math.ceil(period / values.size))
        values = np.tile(values, repeats)
    weekly = values[:period]
    fft = np.fft.fft(weekly)
    features: list[float] = []
    for index in frequency_bins:
        coeff = fft[index]
        features.extend([float(np.real(coeff)), float(np.imag(coeff))])
    return np.asarray(features, dtype=np.float32)


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def make_fiwv(model: nn.Module) -> dict[str, torch.Tensor]:
    return clone_state_dict(model.state_dict())


def fit_mltp_base_learner(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    initial_state: dict[str, torch.Tensor],
) -> tuple[PaperMLTPLSTMRegressor, dict[str, Any]]:
    model = PaperMLTPLSTMRegressor(input_size=int(train_inputs.shape[2])).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_loader, val_loader = _make_loaders(
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        batch_size=batch_size,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = float(val_metrics["MAE"])
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = clone_state_dict(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("ML-TP training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_val_mae": best_val_mae, "best_epoch": best_epoch, "history": history}


@torch.no_grad()
def evaluate_initial_mae(
    state_dict: dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model = PaperMLTPLSTMRegressor(input_size=int(inputs.shape[2])).to(device)
    model.load_state_dict(copy.deepcopy(state_dict))
    metrics = _evaluate_model(model, inputs, targets, device=device, batch_size=batch_size)
    return float(metrics["MAE"])


def select_mltp_initial_state(
    target_meta_feature: np.ndarray,
    meta_samples: list[MLTPSample],
    *,
    target_finetune_inputs: np.ndarray,
    target_finetune_targets: np.ndarray,
    device: torch.device,
    eval_batch_size: int,
    k_neighbors: int,
) -> tuple[MLTPSample, list[dict[str, Any]]]:
    rows = []
    for sample in meta_samples:
        distance = float(np.linalg.norm(target_meta_feature - sample.meta_feature))
        rows.append({"cell_id": int(sample.cell_id), "distance": distance, "sample": sample})
    rows.sort(key=lambda row: row["distance"])
    candidates = rows[: max(1, min(k_neighbors, len(rows)))]

    best_sample: MLTPSample | None = None
    best_initial_mae = float("inf")
    candidate_rows: list[dict[str, Any]] = []
    for row in candidates:
        initial_mae = evaluate_initial_mae(
            row["sample"].state_dict,
            target_finetune_inputs,
            target_finetune_targets,
            device=device,
            batch_size=eval_batch_size,
        )
        candidate_rows.append(
            {
                "candidate_cell_id": int(row["cell_id"]),
                "distance": float(row["distance"]),
                "initial_MAE": float(initial_mae),
            }
        )
        if initial_mae < best_initial_mae:
            best_initial_mae = initial_mae
            best_sample = row["sample"]
    if best_sample is None:
        raise RuntimeError("failed to select ML-TP initial state")
    return best_sample, candidate_rows
