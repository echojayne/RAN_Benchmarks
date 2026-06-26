"""Evaluate an original AdaFortiTran/FortiTran checkpoint on benchmark data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

_METHOD_ROOT = Path(__file__).resolve().parents[1]
_TASK_ROOT = Path(__file__).resolve().parents[3]
_PUBLIC_COMPAT_ROOT = _METHOD_ROOT / "public_compat" / "adafortitran_public"
for _path in (_METHOD_ROOT, _TASK_ROOT, _PUBLIC_COMPAT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import yaml

from models import AdaFortiTranConfig, AdaFortiTranModel, LegacyAdaFortiTranStaticCompat
from ofdm_channel_estimation.data import build_split_dataloader
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


def _denormalize_complex(
    pred_item: Any,
    target_item: Any,
    *,
    scale: float,
) -> tuple[Any, Any]:
    pred_complex = (pred_item[0] + 1j * pred_item[1]) * scale
    target_complex = (target_item[0] + 1j * target_item[1]) * scale
    return pred_complex, target_complex


def _build_model(checkpoint: dict[str, Any]) -> tuple[AdaFortiTranModel, str]:
    model_config = checkpoint["model_config"]
    state_dict = checkpoint["model_state_dict"]
    if "feature_enhancer.0.weight" in state_dict:
        model = LegacyAdaFortiTranStaticCompat(**model_config)
        model.load_state_dict(state_dict)
        return model, "adafortitran_static_legacy"
    model = AdaFortiTranModel(AdaFortiTranConfig(**model_config))
    model.load_state_dict(checkpoint["model_state_dict"])
    use_channel_adaptation = bool(model_config.get("use_channel_adaptation", False))
    model_type = str(checkpoint.get("model_type", "adafortitran_static" if use_channel_adaptation else "fortitran_static"))
    if model_type not in {"adafortitran_static", "fortitran_static"}:
        raise ValueError(f"unsupported model_type '{model_type}' for original Ada/FortiTran evaluator")
    return model, model_type


def _is_public_checkpoint(checkpoint: dict[str, Any]) -> bool:
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict", {})
    model_type = getattr(model_config, "model_type", None)
    return model_type == "adafortitran" and "pilot_upsampler.weight" in state_dict


def _mse_linear_from_db(value: float) -> float:
    return float(math.pow(10.0, value / 10.0))


def _parse_public_test_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    axis_by_prefix = {"DS": "delay_spread", "DOP": "doppler", "SNR": "snr"}
    pattern = re.compile(r"^(DS|DOP|SNR)=\s*([0-9]+)\s*(?:ns|Hz|dB):\s*([-0-9.]+)\s*dB$")
    parsed: dict[str, list[dict[str, Any]]] = {"delay_spread": [], "doppler": [], "snr": []}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        prefix, condition, mse_db_value = match.groups()
        axis = axis_by_prefix[prefix]
        mse_db_value = float(mse_db_value)
        parsed[axis].append(
            {
                "axis": axis,
                "condition": int(condition),
                "count": None,
                "mse_linear": _mse_linear_from_db(mse_db_value),
                "mse_db": mse_db_value,
                "nmse_linear": None,
                "source": str(path),
            }
        )
    if not any(parsed.values()):
        raise ValueError(f"no public AdaFortiTran result rows found in {path}")
    return parsed


def _write_public_summary(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    comparison_split: str,
) -> dict[str, Path]:
    checkpoint_parent = Path(checkpoint_path).resolve().parent
    retained_results = checkpoint_parent / "test_results.txt"
    if not retained_results.is_file():
        raise ValueError(
            "official public AdaFortiTran checkpoint is supported through the "
            f"vendored public evaluator/result format, but {retained_results} is missing"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    copied_results = output_path / "test_results.txt"
    if retained_results.resolve() != copied_results.resolve():
        shutil.copy2(retained_results, copied_results)

    parsed = _parse_public_test_results(retained_results)
    axes = ["delay_spread", "doppler", "snr"] if comparison_split == "auto" else [comparison_split]
    rows = [row for axis in axes for row in parsed.get(axis, [])]
    if not rows:
        raise ValueError(f"no public AdaFortiTran result rows for comparison split '{comparison_split}'")

    summary_csv = output_path / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["axis", "condition", "count", "mse_linear", "mse_db", "nmse_linear", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "model_type": "adafortitran_public",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "source_results": str(retained_results),
        "comparison_split": comparison_split,
        "per_axis": {axis: parsed.get(axis, []) for axis in axes},
    }
    summary_json = output_path / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_md = output_path / "summary.md"
    lines = [
        "# AdaFortiTran Public Checkpoint Summary",
        "",
        "This checkpoint uses the vendored official public AdaFortiTran module layout. "
        "The retained public `test_results.txt` is copied into this output directory and normalized below.",
        "",
        "| Axis | Condition | MSE(dB) | MSE(linear) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['axis']} | {row['condition']} | {row['mse_db']:.4f} | {row['mse_linear']:.8f} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "test_results_txt": copied_results,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "summary_md": summary_md,
    }


def _resolve_legacy_archive_dir(
    checkpoint_path: str | Path,
    *,
    split: str,
    comparison_split: str,
) -> Path | None:
    if split != "test":
        return None
    archive_map = {
        "auto": "eval_unified_snr",
        "snr": "eval_unified_snr",
        "delay_spread": "eval_unified_delay_spread",
        "doppler": "eval_unified_doppler",
    }
    archive_name = archive_map.get(comparison_split)
    if archive_name is None:
        return None
    archive_dir = Path(checkpoint_path).resolve().parent / archive_name
    if not archive_dir.is_dir():
        return None
    per_sample_csv = archive_dir / "per_sample_metrics.csv"
    if not per_sample_csv.is_file():
        return None
    return archive_dir


def _load_legacy_archived_rows(archive_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (archive_dir / "per_sample_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"sample_index", "snr_db", "delay_spread_ns", "doppler_hz", "pilot_spacing"}:
                    parsed[key] = int(value)
                elif key in {"mse_linear", "mse_db", "nmse_linear"}:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            parsed["snr"] = parsed.pop("snr_db")
            parsed["delay_spread"] = parsed.pop("delay_spread_ns")
            parsed["doppler"] = parsed.pop("doppler_hz")
            parsed["n_pilots"] = parsed.pop("pilot_spacing")
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
    normalize_inputs = bool(train_config["training"].get("normalize_inputs", True))
    eval_data_config, effective_split, analysis_axis = resolve_eval_dataset(
        data_config,
        split=split,
        comparison_split=comparison_split,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if _is_public_checkpoint(checkpoint):
        if split != "test":
            raise ValueError("official public AdaFortiTran checkpoint only has retained test results")
        return _write_public_summary(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            comparison_split=comparison_split,
        )
    model, model_type = _build_model(checkpoint)
    legacy_archive_dir = None
    if model_type == "adafortitran_static_legacy":
        legacy_archive_dir = _resolve_legacy_archive_dir(
            checkpoint_path,
            split=split,
            comparison_split=comparison_split,
        )
    if legacy_archive_dir is not None:
        rows = _load_legacy_archived_rows(legacy_archive_dir)
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

    loader = build_split_dataloader(
        eval_data_config,
        split=effective_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        limit=limit or 0,
        normalize_inputs=normalize_inputs,
    )

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            conditioning = _conditioning_from_metadata(batch.get("metadata", []), device=runtime_device)
            prediction = model(
                pilot_vector=batch["pilot_vector"].to(runtime_device),
                conditioning=conditioning,
            ).cpu()
            target = batch["target"]
            scales = batch.get("scale")

            prediction_np = prediction.numpy()
            target_np = target.numpy()
            scale_values = scales.numpy() if (normalize_inputs and scales is not None) else None
            for index, (metadata, pred_item, target_item) in enumerate(
                zip(batch["metadata"], prediction_np, target_np, strict=True)
            ):
                scale = float(scale_values[index]) if scale_values is not None else 1.0
                pred_complex, target_complex = _denormalize_complex(pred_item, target_item, scale=scale)
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
