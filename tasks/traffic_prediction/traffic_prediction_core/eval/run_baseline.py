"""Evaluate an original traffic-forecasting baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_prediction_core.baselines import last_value_forecast, seasonal_naive_forecast
from traffic_prediction_core.eval.common import (
    build_eval_loader,
    denormalize_batch,
    load_normalization_stats,
    resolve_device,
    write_eval_outputs,
)
from traffic_prediction_core.train.common import load_yaml
from traffic_prediction_core.train.train_baseline import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _build_trainable_baseline(config: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[torch.nn.Module | None, str]:
    model, model_name = build_model(config)
    if model is None:
        return None, model_name
    if str(checkpoint.get("model_name", model_name)).strip().lower() != model_name:
        raise ValueError("checkpoint model_name does not match train config")
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_name


def _predict_naive(model_name: str, batch: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    horizon = int(batch["targets"].shape[1])
    if model_name == "naive":
        return last_value_forecast(batch["inputs"], horizon=horizon)
    if model_name == "seasonal_naive":
        seasonal_period = int(config["model"].get("seasonal_period", 24))
        return seasonal_naive_forecast(batch["inputs"], horizon=horizon, seasonal_period=seasonal_period)
    raise ValueError(f"unsupported non-trainable baseline '{model_name}'")


def evaluate(
    *,
    train_config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path,
    batch_size: int,
    num_workers: int,
    limit: int,
    device: str,
) -> dict[str, Path]:
    config = load_yaml(train_config_path)
    checkpoint = Path(checkpoint_path)
    checkpoint_payload = {}
    if checkpoint.suffix == ".pt":
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        checkpoint_payload = load_yaml(checkpoint) if checkpoint.suffix in {".yaml", ".yml"} else {}
        if checkpoint.name.endswith(".json"):
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))

    model, model_name = _build_trainable_baseline(config, checkpoint_payload)
    runtime_device = resolve_device(device)
    if model is not None:
        model = model.to(runtime_device)
        model.eval()

    mean, std = load_normalization_stats(config)
    loader = build_eval_loader(
        config,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        limit=limit,
    )

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            if model is None:
                normalized_prediction = _predict_naive(model_name, batch, config)
            else:
                inputs = batch["inputs"].to(runtime_device, dtype=torch.float32, non_blocking=True)
                normalized_prediction = model(inputs).cpu()
            prediction_raw, target_raw = denormalize_batch(
                normalized_prediction,
                batch["targets"],
                mean=mean,
                std=std,
            )
            predictions.append(prediction_raw)
            targets.append(target_raw)
            metadata.extend(batch["metadata"])

    return write_eval_outputs(
        output_dir,
        model_name=model_name,
        split=split,
        predictions=np.concatenate(predictions, axis=0),
        targets=np.concatenate(targets, axis=0),
        metadata=metadata,
    )


def main() -> int:
    args = parse_args()
    outputs = evaluate(
        train_config_path=args.train_config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        device=args.device,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
