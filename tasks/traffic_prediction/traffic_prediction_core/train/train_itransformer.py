"""Train the traffic-forecasting iTransformer model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from traffic_prediction_core.data import build_split_dataloader, load_manifest
from traffic_prediction_core.models import ITransformerConfig, ITransformerForecaster
from traffic_prediction_core.train.common import load_yaml, resolve_device, run_epoch, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_model(config: dict[str, Any]) -> ITransformerForecaster:
    manifest = load_manifest(config)
    model_cfg = config["model"]
    return ITransformerForecaster(
        ITransformerConfig(
            num_regions=int(manifest["top_k_regions"]),
            context_len=int(manifest["context_len"]),
            horizon=int(manifest["horizon"]),
            d_model=int(model_cfg.get("d_model", 128)),
            depth=int(model_cfg.get("depth", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=int(model_cfg.get("ffn_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training_cfg = config["training"]
    if args.output_dir:
        training_cfg["output_dir"] = args.output_dir
    if args.epochs > 0:
        training_cfg["epochs"] = args.epochs

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(training_cfg.get("seed", 13)))
    model = build_model(config)
    device = resolve_device(str(training_cfg.get("device", "auto")))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    train_loader = build_split_dataloader(
        config,
        split="train",
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        limit=args.train_limit,
        normalize=True,
    )
    val_loader = build_split_dataloader(
        config,
        split="val",
        batch_size=int(training_cfg.get("batch_size", 64)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        limit=args.val_limit,
        normalize=True,
        shuffle=False,
    )

    epochs = int(training_cfg.get("epochs", 20))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
    best_val_mae = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device=device, optimizer=optimizer, grad_clip_norm=grad_clip_norm)
        val_metrics = run_epoch(model, val_loader, device=device, optimizer=None, grad_clip_norm=grad_clip_norm)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": "itransformer",
                    "state_dict": model.state_dict(),
                    "config": config,
                    "best_val_mae": best_val_mae,
                    "best_epoch": best_epoch,
                },
                output_dir / "best.pt",
            )

    save_json(
        output_dir / "train_history.json",
        {
            "model_name": "itransformer",
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "history": history,
        },
    )
    print(json.dumps({"model": "itransformer", "best_val_mae": best_val_mae, "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
