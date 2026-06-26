"""Train the original AdaFortiTran/FortiTran baseline."""

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

from models import AdaFortiTranConfig, AdaFortiTranModel
from ofdm_channel_estimation.data import build_split_dataloader


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


def _conditioning_from_metadata(
    metadata_list: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not metadata_list:
        return None
    snr = torch.tensor([float(item["snr"]) for item in metadata_list], device=device, dtype=torch.float32)
    delay_spread = torch.tensor([float(item["delay_spread"]) for item in metadata_list], device=device, dtype=torch.float32)
    doppler = torch.tensor([float(item["doppler"]) for item in metadata_list], device=device, dtype=torch.float32)
    return snr, delay_spread, doppler


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError(f"unsupported checkpoint format: {path}")


def _build_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_subcarriers": int(model_cfg.get("num_subcarriers", 120)),
        "num_symbols": int(model_cfg.get("num_symbols", 14)),
        "input_channels": int(model_cfg.get("input_channels", 4)),
        "output_channels": int(model_cfg.get("output_channels", 2)),
        "pilot_vector_length": int(model_cfg.get("pilot_vector_length", 80)),
        "d_enc": int(model_cfg.get("d_enc", 32)),
        "encoder_layers": int(model_cfg.get("encoder_layers", 6)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "ffn_dim": int(model_cfg.get("ffn_dim", 64)),
        "activation": str(model_cfg.get("activation", "gelu")),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "max_seq_len": int(model_cfg.get("max_seq_len", 512)),
        "pos_encoding_type": str(model_cfg.get("pos_encoding_type", "learnable")),
        "patch_subcarriers": int(model_cfg.get("patch_subcarriers", 3)),
        "patch_symbols": int(model_cfg.get("patch_symbols", 2)),
        "shallow_channels": int(model_cfg.get("shallow_channels", 8)),
        "hidden_channels": int(model_cfg.get("hidden_channels", 32)),
        "use_channel_adaptation": bool(model_cfg.get("use_channel_adaptation", False)),
        "channel_adaptivity_hidden_sizes": tuple(
            int(value) for value in model_cfg.get("channel_adaptivity_hidden_sizes", [7, 42, 560])
        )
        if bool(model_cfg.get("use_channel_adaptation", False))
        else None,
        "adaptive_token_length": (
            int(model_cfg["adaptive_token_length"])
            if model_cfg.get("adaptive_token_length") is not None
            else None
        ),
    }


def build_model(train_cfg: dict[str, Any]) -> tuple[nn.Module, dict[str, Any], str]:
    model_config = _build_model_config(train_cfg["model"])
    model = AdaFortiTranModel(AdaFortiTranConfig(**model_config))
    model_type = "adafortitran_static" if model_config["use_channel_adaptation"] else "fortitran_static"
    return model, model_config, model_type


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
    nmse_total = 0.0
    sample_count = 0
    criterion = nn.MSELoss()
    for batch in loader:
        pilot_vector = batch["pilot_vector"].to(device=device, dtype=torch.float32, non_blocking=True)
        target = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        conditioning = _conditioning_from_metadata(batch.get("metadata", []), device=device)
        with torch.set_grad_enabled(is_train):
            prediction = model(pilot_vector=pilot_vector, conditioning=conditioning)
            loss = criterion(prediction, target)
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
    data_config = load_data_config(train_cfg, train_config_path)
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
    set_seed(int(training_cfg.get("seed", 7)))
    device = resolve_device(str(training_cfg.get("device", "auto")))

    model, model_config, model_type = build_model(train_cfg)
    if args.init_checkpoint:
        model.load_state_dict(_load_state_dict(args.init_checkpoint), strict=False)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = None
    lr_gamma = float(training_cfg.get("lr_gamma", 1.0))
    if lr_gamma > 0.0 and abs(lr_gamma - 1.0) > 1e-12:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    normalize_inputs = bool(training_cfg.get("normalize_inputs", True))
    batch_size = int(training_cfg.get("batch_size", 512))
    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = build_split_dataloader(
        data_config,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        limit=args.train_limit,
        normalize_inputs=normalize_inputs,
    )
    val_loader = build_split_dataloader(
        data_config,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=args.val_limit,
        normalize_inputs=normalize_inputs,
    )

    epochs = int(training_cfg.get("epochs", 100))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
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
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            grad_clip_norm=grad_clip_norm,
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
                    "model_type": model_type,
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_nmse": best_val_nmse,
                    "epoch": best_epoch,
                    "config": train_cfg,
                    "train_config_path": str(Path(args.config).resolve()),
                },
                output_dir / checkpoint_name,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        if scheduler is not None:
            scheduler.step()

    save_json(
        output_dir / "train_history.json",
        {
            "model_name": model_type,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_nmse": best_val_nmse,
            "history": history,
        },
    )
    print(
        json.dumps(
            {
                "model": model_type,
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
