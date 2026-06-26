"""Naive forecasting baselines."""

from __future__ import annotations

import torch


def last_value_forecast(inputs: torch.Tensor, *, horizon: int) -> torch.Tensor:
    last_step = inputs[:, -1:, :]
    return last_step.repeat(1, horizon, 1)


def naive_last_value(inputs: torch.Tensor, *, horizon: int) -> torch.Tensor:
    return last_value_forecast(inputs, horizon=horizon)


def seasonal_naive_forecast(inputs: torch.Tensor, *, horizon: int, seasonal_period: int = 24) -> torch.Tensor:
    if inputs.shape[1] < seasonal_period:
        return last_value_forecast(inputs, horizon=horizon)
    parts = []
    for step in range(horizon):
        index = step % seasonal_period
        parts.append(inputs[:, index : index + 1, :])
    return torch.cat(parts, dim=1)


def seasonal_naive(inputs: torch.Tensor, *, horizon: int) -> torch.Tensor:
    return seasonal_naive_forecast(inputs, horizon=horizon)
