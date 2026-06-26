"""Common training helpers for traffic forecasting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def batch_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def batch_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def run_epoch(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_total = 0.0
    mae_total = 0.0
    mse_total = 0.0
    sample_count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["targets"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            pred = model(inputs)
            loss = batch_mse(pred, targets)
            mae = batch_mae(pred, targets)
            mse = batch_mse(pred, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        batch_size = int(inputs.shape[0])
        loss_total += float(loss.item()) * batch_size
        mae_total += float(mae.item()) * batch_size
        mse_total += float(mse.item()) * batch_size
        sample_count += batch_size
    return {
        "loss": loss_total / max(sample_count, 1),
        "mae": mae_total / max(sample_count, 1),
        "rmse": float(np.sqrt(mse_total / max(sample_count, 1))),
        "mse": mse_total / max(sample_count, 1),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
