"""Evaluate an original iTransformer checkpoint on one data split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_prediction_core.data import load_manifest
from traffic_prediction_core.eval.common import (
    build_eval_loader,
    denormalize_batch,
    load_normalization_stats,
    resolve_device,
    write_eval_outputs,
)
from traffic_prediction_core.train.common import load_yaml
from traffic_prediction_core.train.train_itransformer import build_model


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


def _resolve_config(train_config_path: str | Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = load_yaml(train_config_path)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        return config
    if "model" not in config:
        config["model"] = checkpoint_config.get("model", {})
    elif not config["model"]:
        config["model"] = checkpoint_config.get("model", {})
    return config


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = str(checkpoint.get("model_name", "itransformer")).strip().lower()
    if model_name != "itransformer":
        raise ValueError(f"unsupported model_name '{model_name}' for original iTransformer evaluator")

    config = _resolve_config(train_config_path, checkpoint)
    load_manifest(config)
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])

    runtime_device = resolve_device(device)
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
            inputs = batch["inputs"].to(runtime_device, dtype=torch.float32, non_blocking=True)
            normalized_prediction = model(inputs)
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
        model_name="itransformer",
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
