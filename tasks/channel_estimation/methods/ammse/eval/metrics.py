"""Evaluation metrics for channel-estimation baselines."""

from __future__ import annotations

import numpy as np


def mse_linear(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    return float(np.mean(np.abs(pred_arr - target_arr) ** 2))


def mse_db(pred: np.ndarray, target: np.ndarray) -> float:
    return float(10.0 * np.log10(mse_linear(pred, target) + 1e-12))


def nmse_linear(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    numerator = np.mean(np.abs(pred_arr - target_arr) ** 2)
    denominator = np.mean(np.abs(target_arr) ** 2) + 1e-12
    return float(numerator / denominator)
