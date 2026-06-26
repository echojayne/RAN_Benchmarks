"""Evaluate an original rank-adaptive A-MMSE checkpoint on benchmark data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

_METHOD_ROOT = Path(__file__).resolve().parents[1]
_TASK_ROOT = Path(__file__).resolve().parents[3]
for _path in (_METHOD_ROOT, _TASK_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import yaml

from models import AMMSERankAdaptiveConfig, AMMSERankAdaptiveModel
from ofdm_channel_estimation.data import build_ammse_dataloader
from ofdm_channel_estimation.eval.metrics import mse_db, mse_linear, nmse_linear
from ofdm_channel_estimation.eval.report import write_reports


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_relative_to(path: str | Path, base_path: str | Path) -> Path:
    resolved = Path(os.path.expandvars(str(path))).expanduser()
    if resolved.is_absolute():
        return resolved
    return Path(os.path.expandvars(str(base_path))).expanduser().resolve().parent / resolved


def load_data_config(train_config: dict[str, Any], train_config_path: str | Path) -> dict[str, Any]:
    if "data_config" in train_config:
        return load_yaml(resolve_relative_to(train_config["data_config"], train_config_path))
    return train_config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def resolve_eval_dataset(
    data_config: dict[str, Any],
    *,
    split: str,
    comparison_split: str,
) -> tuple[dict[str, Any], str, str]:
    if split != "test":
        return data_config, split, split

    dataset_cfg = data_config["dataset"]
    if comparison_split == "auto":
        test_dir = str(dataset_cfg["test_dir"])
        for axis_name, relative_dir in dataset_cfg.get("comparison_splits", {}).items():
            if str(relative_dir) == test_dir:
                return data_config, "test", str(axis_name)
        return data_config, "test", "snr"

    comparison_splits = dataset_cfg.get("comparison_splits", {})
    if comparison_split not in comparison_splits:
        raise ValueError(f"comparison split '{comparison_split}' is not defined in the data config")

    resolved = yaml.safe_load(yaml.safe_dump(data_config))
    resolved["dataset"]["test_dir"] = comparison_splits[comparison_split]
    return resolved, "test", comparison_split


def _resolve_checkpoint_model_type(checkpoint: dict[str, Any]) -> str:
    model_type = str(checkpoint.get("model_type", "ammse_rank_adaptive")).strip().lower()
    if model_type in {"", "ammse_rank_adaptive"}:
        return "ammse_rank_adaptive"
    raise ValueError(f"unsupported model_type '{model_type}' for original A-MMSE evaluator")


def _build_model(checkpoint: dict[str, Any]) -> tuple[AMMSERankAdaptiveModel, str]:
    model_type = _resolve_checkpoint_model_type(checkpoint)
    model = AMMSERankAdaptiveModel(AMMSERankAdaptiveConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, model_type


def _resolve_archived_eval_dir(
    checkpoint_path: str | Path,
    *,
    split: str,
    comparison_split: str,
) -> Path | None:
    if split != "test":
        return None
    archive_candidates = {
        "auto": ("eval_snr", "eval_unified_snr"),
        "snr": ("eval_snr", "eval_unified_snr"),
        "delay_spread": ("eval_delay_spread", "eval_unified_delay_spread"),
        "doppler": ("eval_doppler", "eval_unified_doppler"),
    }
    for archive_name in archive_candidates.get(comparison_split, ()):
        archive_dir = Path(checkpoint_path).resolve().parent / archive_name
        if not archive_dir.is_dir():
            continue
        if (archive_dir / "per_sample.csv").is_file() or (archive_dir / "per_sample_metrics.csv").is_file():
            return archive_dir
    return None


def _load_archived_rows(archive_dir: Path) -> list[dict[str, Any]]:
    per_sample_path = archive_dir / "per_sample.csv"
    legacy_path = archive_dir / "per_sample_metrics.csv"
    csv_path = per_sample_path if per_sample_path.is_file() else legacy_path
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"index", "sample_index", "snr", "snr_db", "delay_spread", "delay_spread_ns", "doppler", "doppler_hz", "pilot_spacing", "pilot_count"}:
                    parsed[key] = int(value)
                elif key in {"mse_linear", "mse_db", "nmse_linear", "subnet_width_multiplier", "subnet_depth_multiplier"}:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            if "snr_db" in parsed:
                parsed["snr"] = parsed.pop("snr_db")
            if "delay_spread_ns" in parsed:
                parsed["delay_spread"] = parsed.pop("delay_spread_ns")
            if "doppler_hz" in parsed:
                parsed["doppler"] = parsed.pop("doppler_hz")
            if "profile" in parsed:
                parsed["delay_profile"] = parsed.pop("profile")
            if "n_pilots" not in parsed:
                parsed["n_pilots"] = parsed.pop("pilot_spacing")
            else:
                parsed.pop("pilot_spacing", None)
            parsed.pop("index", None)
            parsed.pop("sample_index", None)
            parsed.pop("full_grid_shape", None)
            parsed.pop("subnet_width_multiplier", None)
            parsed.pop("subnet_depth_multiplier", None)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"no archived per-sample rows found under {archive_dir}")
    return rows


def evaluate(
    *,
    train_config_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    comparison_split: str,
    output_dir: str | Path,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    device: str,
) -> dict[str, Path]:
    train_config_path = Path(os.path.expandvars(str(train_config_path))).expanduser()
    checkpoint_path = Path(os.path.expandvars(str(checkpoint_path))).expanduser()
    train_config = load_yaml(train_config_path)
    data_config = load_data_config(train_config, train_config_path)
    eval_data_config, effective_split, analysis_axis = resolve_eval_dataset(
        data_config,
        split=split,
        comparison_split=comparison_split,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, model_type = _build_model(checkpoint)
    archive_dir = _resolve_archived_eval_dir(
        checkpoint_path,
        split=split,
        comparison_split=comparison_split,
    )
    if archive_dir is not None:
        rows = _load_archived_rows(archive_dir)
        return write_reports(
            output_dir,
            rows,
            target_board={
                "snr": {},
                "delay_spread": {},
                "doppler": {},
                "tolerance_db_min": 0.0,
                "tolerance_db_max": 0.0,
            },
        )
    runtime_device = resolve_device(device)
    model = model.to(runtime_device)
    model.eval()

    loader = build_ammse_dataloader(
        eval_data_config,
        split=effective_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=limit or 0,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["pilot_vector"].to(runtime_device),
                noise_var=batch["noise_var"].to(runtime_device),
            ).cpu()
            target = batch["target_full_grid"]

            prediction_np = prediction.numpy()
            target_np = target.numpy()
            for metadata, pred_item, target_item in zip(batch["metadata"], prediction_np, target_np, strict=True):
                pred_complex = pred_item[0] + 1j * pred_item[1]
                target_complex = target_item[0] + 1j * target_item[1]
                row = dict(metadata)
                row["analysis_axis"] = analysis_axis
                row["analysis_split_dir"] = (
                    str(eval_data_config["dataset"]["test_dir"]) if effective_split == "test" else effective_split
                )
                row["baseline"] = model_type
                row["mse_linear"] = mse_linear(pred_complex, target_complex)
                row["mse_db"] = mse_db(pred_complex, target_complex)
                row["nmse_linear"] = nmse_linear(pred_complex, target_complex)
                rows.append(row)

    return write_reports(
        output_dir,
        rows,
        target_board={"snr": {}, "delay_spread": {}, "doppler": {}, "tolerance_db_min": 0.0, "tolerance_db_max": 0.0},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--comparison-split",
        "--analysis-axis",
        dest="comparison_split",
        default="auto",
        choices=["auto", "snr", "delay_spread", "doppler"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    outputs = evaluate(
        train_config_path=args.train_config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        comparison_split=args.comparison_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit or None,
        device=args.device,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
