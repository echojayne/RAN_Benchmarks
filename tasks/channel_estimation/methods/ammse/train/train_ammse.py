"""Train the original rank-adaptive A-MMSE baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_METHOD_ROOT = Path(__file__).resolve().parents[1]
_TASK_ROOT = Path(__file__).resolve().parents[3]
for _path in (_METHOD_ROOT, _TASK_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
import yaml
from torch import nn

from models import AMMSERankAdaptiveConfig, AMMSERankAdaptiveModel
from ofdm_channel_estimation.data import build_ammse_dataloader
from ofdm_channel_estimation.data.adafortitran_dataset import _pilot_mask_from_config


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_relative_to(path: str | Path, base_path: str | Path) -> Path:
    resolved = Path(os.path.expandvars(str(path))).expanduser()
    if resolved.is_absolute():
        return resolved
    return Path(os.path.expandvars(str(base_path))).expanduser().resolve().parent / resolved


def load_data_config(train_cfg: dict[str, Any], train_config_path: str | Path) -> dict[str, Any]:
    if "data_config" in train_cfg:
        return load_yaml(resolve_relative_to(train_cfg["data_config"], train_config_path))
    return train_cfg


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def batch_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target) ** 2, dim=dims)
    denominator = torch.sum(target**2, dim=dims).clamp_min(1e-12)
    return torch.mean(numerator / denominator)


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError(f"unsupported checkpoint format: {path}")


def build_model_config(train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = train_cfg["model"]
    pilot_mask = _pilot_mask_from_config(data_cfg)
    pilot_subcarrier_tokens = int(pilot_mask.any(axis=1).sum())
    pilot_symbol_tokens = int(pilot_mask.any(axis=0).sum())
    pilot_vector_length = int(pilot_mask.sum())
    return {
        "num_subcarriers": int(model_cfg.get("num_subcarriers", pilot_mask.shape[0])),
        "num_symbols": int(model_cfg.get("num_symbols", pilot_mask.shape[1])),
        "pilot_vector_length": int(model_cfg.get("pilot_vector_length", pilot_vector_length)),
        "pilot_subcarrier_tokens": int(model_cfg.get("pilot_subcarrier_tokens", pilot_subcarrier_tokens)),
        "pilot_symbol_tokens": int(model_cfg.get("pilot_symbol_tokens", pilot_symbol_tokens)),
        "d_model": int(model_cfg.get("d_model", 64)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "frequency_layers": int(model_cfg.get("frequency_layers", 2)),
        "temporal_layers": int(model_cfg.get("temporal_layers", 2)),
        "ffn_dim": int(model_cfg.get("ffn_dim", 128)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "filter_rank": int(model_cfg.get("filter_rank", 8)),
        "noise_embed_dim": int(model_cfg.get("noise_embed_dim", 32)),
    }


def build_model(train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    architecture = str(train_cfg["model"].get("architecture", "rank_adaptive")).strip().lower()
    if architecture not in {"rank_adaptive", "rank_adaptive_static", "ra_a_mmse", "paper_strict"}:
        raise ValueError(f"unsupported A-MMSE architecture '{architecture}'")
    model_config = build_model_config(train_cfg, data_cfg)
    return AMMSERankAdaptiveModel(AMMSERankAdaptiveConfig(**model_config)), model_config


def build_loss_fn(training_cfg: dict[str, Any]) -> nn.Module:
    loss_name = str(training_cfg.get("loss", "mse")).strip().lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=float(training_cfg.get("huber_delta", 0.1)))
    raise ValueError(f"unsupported loss '{loss_name}'")


def build_optimizer(model: nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).strip().lower()
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "nadam":
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer '{optimizer_name}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(training_cfg.get("scheduler", "exponential")).strip().lower()
    if scheduler_name in {"", "none"}:
        return None
    if scheduler_name == "exponential":
        gamma = float(training_cfg.get("lr_gamma", 1.0))
        if abs(gamma - 1.0) < 1e-12:
            return None
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_cfg.get("plateau_factor", 0.7)),
            patience=int(training_cfg.get("plateau_patience", 4)),
            min_lr=float(training_cfg.get("min_learning_rate", 1.0e-14)),
        )
    raise ValueError(f"unsupported scheduler '{scheduler_name}'")


def run_epoch(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
    loss_fn: nn.Module,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_total = 0.0
    nmse_total = 0.0
    sample_count = 0
    for batch in loader:
        pilot_vector = batch["pilot_vector"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target_full_grid"].to(device=device, dtype=torch.float32, non_blocking=True)
        noise_var = batch["noise_var"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            prediction = model(pilot_vector, noise_var=noise_var)
            loss = loss_fn(prediction, target)
            nmse = batch_nmse(prediction, target)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        batch_size = int(target.shape[0])
        loss_total += float(loss.item()) * batch_size
        nmse_total += float(nmse.item()) * batch_size
        sample_count += batch_size
    return {
        "loss": loss_total / max(sample_count, 1),
        "nmse": nmse_total / max(sample_count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--init-checkpoint", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_config_path = Path(os.path.expandvars(str(args.config))).expanduser()
    train_cfg = load_yaml(train_config_path)
    data_cfg = load_data_config(train_cfg, train_config_path)
    training_cfg = train_cfg["training"]

    if args.output_dir:
        training_cfg["output_dir"] = args.output_dir
    if args.batch_size > 0:
        training_cfg["batch_size"] = args.batch_size
    if args.epochs > 0:
        training_cfg["epochs"] = args.epochs
    if args.num_workers >= 0:
        training_cfg["num_workers"] = args.num_workers
    if args.device:
        training_cfg["device"] = args.device

    output_dir = Path(os.path.expandvars(str(training_cfg["output_dir"]))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(training_cfg.get("seed", 13)))
    device = resolve_device(str(training_cfg.get("device", "auto")))

    model, model_config = build_model(train_cfg, data_cfg)
    if args.init_checkpoint:
        model.load_state_dict(_load_state_dict(args.init_checkpoint), strict=False)
    model = model.to(device)
    loss_fn = build_loss_fn(training_cfg)
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg)

    batch_size = int(training_cfg.get("batch_size", 12288))
    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = build_ammse_dataloader(
        data_cfg,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        limit=args.train_limit,
    )
    val_loader = build_ammse_dataloader(
        data_cfg,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=args.val_limit,
    )

    epochs = int(training_cfg.get("epochs", 100))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))
    patience = int(training_cfg.get("early_stopping_patience", 10))
    checkpoint_name = str(training_cfg.get("checkpoint_name", "best.pt"))

    best_val_loss = float("inf")
    best_val_nmse = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            loss_fn=loss_fn,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            grad_clip_norm=grad_clip_norm,
            loss_fn=loss_fn,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_nmse": train_metrics["nmse"],
                "val_loss": val_metrics["loss"],
                "val_nmse": val_metrics["nmse"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_nmse = val_metrics["nmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_type": "ammse_rank_adaptive",
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_nmse": best_val_nmse,
                    "epoch": best_epoch,
                    "train_config_path": str(train_config_path.resolve()),
                    "config": train_cfg,
                },
                output_dir / checkpoint_name,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

    save_json(
        output_dir / "train_history.json",
        {
            "model_name": "ammse_rank_adaptive",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_nmse": best_val_nmse,
            "history": history,
        },
    )
    print(
        json.dumps(
            {
                "model": "ammse_rank_adaptive",
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_nmse": best_val_nmse,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
