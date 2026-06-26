"""Prepare the Telecom Italia Milan dataset for traffic forecasting."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_COLUMNS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
]

RAW_SUFFIXES = {".txt", ".csv", ".tsv", ".parquet", ".pq"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_column_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _dataset_cfg(config: dict[str, Any]) -> dict[str, Any]:
    if "dataset" in config:
        return config["dataset"]
    raise ValueError("prepare config must define a top-level 'dataset' section")


def _prepared_dir(dataset_cfg: dict[str, Any]) -> Path:
    path = dataset_cfg.get("output_dir", dataset_cfg.get("prepared_dir"))
    if path is None:
        raise ValueError("dataset config must define output_dir or prepared_dir")
    return Path(os.path.expandvars(str(path))).expanduser()


def _raw_dir(dataset_cfg: dict[str, Any]) -> Path:
    raw_dir = dataset_cfg.get("raw_dir")
    if raw_dir is None:
        raise ValueError("dataset config must define raw_dir")
    return Path(os.path.expandvars(str(raw_dir))).expanduser()


def _train_ratio(dataset_cfg: dict[str, Any]) -> float:
    return float(dataset_cfg.get("train_ratio", dataset_cfg.get("train_fraction", 0.7)))


def _val_ratio(dataset_cfg: dict[str, Any]) -> float:
    return float(dataset_cfg.get("val_ratio", dataset_cfg.get("val_fraction", 0.1)))


def _explicit_split_time(dataset_cfg: dict[str, Any], key: str) -> pd.Timestamp | None:
    value = dataset_cfg.get(key)
    if value in (None, ""):
        return None
    return pd.Timestamp(value, tz="UTC")


def _high_quantile(dataset_cfg: dict[str, Any]) -> float:
    return float(dataset_cfg.get("high_load_quantile", dataset_cfg.get("high_quantile", 0.8)))


def _low_quantile(dataset_cfg: dict[str, Any]) -> float:
    return float(dataset_cfg.get("low_load_quantile", dataset_cfg.get("low_quantile", 0.2)))


def _file_pattern(dataset_cfg: dict[str, Any]) -> str:
    return str(dataset_cfg.get("file_pattern", "*"))


def _target_column(dataset_cfg: dict[str, Any]) -> str:
    return str(dataset_cfg.get("target_field", "internet_traffic"))


def _timestamp_column(dataset_cfg: dict[str, Any]) -> str:
    return str(dataset_cfg.get("timestamp_column", dataset_cfg.get("time_column", "time_interval")))


def _region_column(dataset_cfg: dict[str, Any]) -> str:
    return str(dataset_cfg.get("region_column", "square_id"))


def _build_smoke_frame(*, num_regions: int, num_steps_10m: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=num_steps_10m, freq="10min", tz="UTC")
    rows = []
    for region_id in range(1, num_regions + 1):
        scale = rng.uniform(50.0, 120.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for step, timestamp in enumerate(timestamps):
            day_wave = 1.0 + 0.5 * np.sin((step % 144) / 144.0 * 2.0 * np.pi + phase)
            rush_bonus = 18.0 if step % 144 in set(range(42, 49)) | set(range(96, 103)) else 0.0
            noise = rng.normal(0.0, 2.0)
            rows.append(
                {
                    "square_id": region_id,
                    "time_interval": int(timestamp.value // 10**6),
                    "country_code": 39,
                    "sms_in": 0.0,
                    "sms_out": 0.0,
                    "call_in": 0.0,
                    "call_out": 0.0,
                    "internet_traffic": max(0.0, scale * day_wave + rush_bonus + noise),
                }
            )
    return pd.DataFrame(rows)


def _read_raw_table(path: Path, column_names: list[str]) -> pd.DataFrame:
    attempts = [
        {"sep": "\t", "header": None},
        {"sep": ",", "header": None},
        {"sep": ";", "header": None},
        {"sep": r"\s+", "header": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ","},
        {"sep": ";"},
    ]
    for kwargs in attempts:
        try:
            frame = pd.read_csv(path, **kwargs)
        except Exception:
            continue
        if frame.empty:
            continue
        normalized = [_normalize_column_name(column) for column in frame.columns]
        if normalized == column_names:
            frame.columns = normalized
            return frame
        if len(frame.columns) == len(column_names):
            frame.columns = column_names
            return frame
        frame.columns = normalized
        if {"square_id", "time_interval", "internet_traffic"}.issubset(frame.columns):
            return frame
    raise ValueError(f"could not parse Milan raw file: {path}")


def _list_raw_sources(dataset_cfg: dict[str, Any]) -> list[Path]:
    raw_dir = _raw_dir(dataset_cfg)
    pattern = _file_pattern(dataset_cfg)
    candidates = sorted(path for path in raw_dir.rglob(pattern) if path.is_file() and path.suffix.lower() in RAW_SUFFIXES)
    if not candidates:
        raise FileNotFoundError(
            f"no raw Milan files found under {raw_dir}. "
            "Place official raw files there manually or enable dataset.smoke."
        )
    return candidates


def _read_source_frame(path: Path, dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        frame.columns = [_normalize_column_name(column) for column in frame.columns]
        return frame
    return _read_raw_table(path, list(dataset_cfg.get("column_names", DEFAULT_COLUMNS)))


def _aggregate_hourly_rows(frame: pd.DataFrame, dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    smoke_cfg = dataset_cfg.get("smoke", {})
    region_col = _region_column(dataset_cfg)
    timestamp_col = _timestamp_column(dataset_cfg)
    target_col = _target_column(dataset_cfg)
    timestamp_unit = str(dataset_cfg.get("timestamp_unit", "ms"))
    aggregate_minutes = int(dataset_cfg.get("aggregate_minutes", 60))

    required = {region_col, timestamp_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw frame is missing required columns: {sorted(missing)}")

    keep = frame[[region_col, timestamp_col, target_col]].copy()
    del frame
    keep[region_col] = pd.to_numeric(keep[region_col], errors="coerce").astype("Int64")
    keep[timestamp_col] = pd.to_numeric(keep[timestamp_col], errors="coerce")
    keep[target_col] = pd.to_numeric(keep[target_col], errors="coerce")
    keep = keep.dropna(subset=[region_col, timestamp_col, target_col]).copy()
    keep[region_col] = keep[region_col].astype(np.int64)
    keep["timestamp"] = pd.to_datetime(keep[timestamp_col].astype(np.int64), unit=timestamp_unit, utc=True)

    # Collapse duplicate `(timestamp, region)` rows before hourly aggregation.
    coarse = keep.groupby(["timestamp", region_col], as_index=False)[target_col].sum()
    coarse["bucket"] = coarse["timestamp"].dt.floor(f"{aggregate_minutes}min")
    return coarse.groupby(["bucket", region_col], as_index=False)[target_col].sum()


def _build_hourly_pivot(dataset_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    smoke_cfg = dataset_cfg.get("smoke", {})
    target_col = _target_column(dataset_cfg)
    region_col = _region_column(dataset_cfg)
    aggregate_minutes = int(dataset_cfg.get("aggregate_minutes", 60))

    if bool(smoke_cfg.get("enabled", False)):
        frame = _build_smoke_frame(
            num_regions=int(smoke_cfg.get("num_regions", 8)),
            num_steps_10m=int(smoke_cfg.get("num_steps_10m", 24 * 6 * 28)),
            seed=int(smoke_cfg.get("seed", 7)),
        )
        hourly = _aggregate_hourly_rows(frame, dataset_cfg)
        raw_sources = ["synthetic_smoke_generator"]
    else:
        candidates = _list_raw_sources(dataset_cfg)
        hourly_parts: list[pd.DataFrame] = []
        for index, path in enumerate(candidates, start=1):
            frame = _read_source_frame(path, dataset_cfg)
            hourly_parts.append(_aggregate_hourly_rows(frame, dataset_cfg))
            print(json.dumps({"stage": "aggregate_source", "index": index, "total": len(candidates), "source": path.name}))
        hourly = (
            pd.concat(hourly_parts, ignore_index=True)
            .groupby(["bucket", region_col], as_index=False)[target_col]
            .sum()
            .sort_values(["bucket", region_col])
        )
        raw_sources = [str(path) for path in candidates]

    pivot = hourly.pivot(index="bucket", columns=region_col, values=target_col).sort_index()
    if pivot.empty:
        raise ValueError("hourly pivot is empty after preprocessing")

    full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq=f"{aggregate_minutes}min", tz="UTC")
    pivot = pivot.reindex(full_index, fill_value=0.0)
    pivot.columns = pivot.columns.astype(np.int64)
    return pivot.sort_index(), raw_sources


def _split_frame(pivot: pd.DataFrame, *, train_ratio: float, val_ratio: float) -> dict[str, pd.DataFrame]:
    # Legacy ratio split kept for smoke or quick experiments.
    total_rows = len(pivot)
    if total_rows < 3:
        raise ValueError("hourly pivot is too short to split into train/val/test")
    train_end = max(1, int(total_rows * train_ratio))
    val_end = max(train_end + 1, train_end + int(total_rows * val_ratio))
    val_end = min(val_end, total_rows - 1)
    return {
        "train": pivot.iloc[:train_end].copy(),
        "val": pivot.iloc[train_end:val_end].copy(),
        "test": pivot.iloc[val_end:].copy(),
    }


def _split_frame_by_time(pivot: pd.DataFrame, dataset_cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    train_end = _explicit_split_time(dataset_cfg, "train_end")
    val_end = _explicit_split_time(dataset_cfg, "val_end")
    test_end = _explicit_split_time(dataset_cfg, "test_end")
    if train_end is None or val_end is None:
        return _split_frame(pivot, train_ratio=_train_ratio(dataset_cfg), val_ratio=_val_ratio(dataset_cfg))

    train = pivot.loc[pivot.index < train_end].copy()
    val = pivot.loc[(pivot.index >= train_end) & (pivot.index < val_end)].copy()
    if test_end is None:
        test = pivot.loc[pivot.index >= val_end].copy()
    else:
        test = pivot.loc[(pivot.index >= val_end) & (pivot.index < test_end)].copy()

    if min(len(train), len(val), len(test)) <= 0:
        raise ValueError(
            "date-based split produced an empty split. "
            "Check train_end/val_end/test_end in the prepare config."
        )
    return {"train": train, "val": val, "test": test}


def _build_windows(
    values: np.ndarray,
    timestamps_ns: np.ndarray,
    *,
    context_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(values) < context_len + horizon:
        raise ValueError("not enough rows to build forecasting windows")
    inputs = []
    targets = []
    target_times = []
    target_start_times = []
    sample_index = []
    num_samples = len(values) - context_len - horizon + 1
    for start in range(num_samples):
        middle = start + context_len
        end = middle + horizon
        inputs.append(values[start:middle])
        targets.append(values[middle:end])
        target_times.append(timestamps_ns[middle:end])
        target_start_times.append(int(timestamps_ns[middle]))
        sample_index.append(start)
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_times, dtype=np.int64),
        np.asarray(target_start_times, dtype=np.int64),
        np.asarray(sample_index, dtype=np.int64),
    )


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = _dataset_cfg(config)
    prepared_dir = _prepared_dir(dataset_cfg)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    hourly, raw_sources = _build_hourly_pivot(dataset_cfg)
    split_frames = _split_frame_by_time(hourly, dataset_cfg)

    top_k = min(int(dataset_cfg.get("top_k_regions", 64)), hourly.shape[1])
    train_scores = split_frames["train"].sum(axis=0).sort_values(ascending=False)
    selected_regions = [int(value) for value in train_scores.head(top_k).index.tolist()]
    selection_scores = {str(int(index)): float(value) for index, value in train_scores.head(top_k).items()}

    split_frames = {name: frame.loc[:, selected_regions].copy() for name, frame in split_frames.items()}
    train_values = split_frames["train"].to_numpy(dtype=np.float32)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    context_len = int(dataset_cfg.get("context_len", 24))
    horizon = int(dataset_cfg.get("horizon", 6))
    outputs: dict[str, str] = {}
    split_meta: dict[str, Any] = {}
    selected_regions_csv = prepared_dir / "selected_regions.csv"
    pd.DataFrame(
        {
            "region_id": selected_regions,
            "train_sum": [selection_scores[str(region)] for region in selected_regions],
            "train_mean_hourly": [
                float(split_frames["train"][region].mean()) for region in selected_regions
            ],
        }
    ).to_csv(selected_regions_csv, index=False)

    train_target_scores = None
    for split_name, frame in split_frames.items():
        values = frame.to_numpy(dtype=np.float32)
        timestamps_ns = frame.index.view("int64").astype(np.int64)
        inputs, targets, target_times_ns, target_start_time_ns, sample_index = _build_windows(
            values,
            timestamps_ns,
            context_len=context_len,
            horizon=horizon,
        )
        split_path = prepared_dir / f"{split_name}.npz"
        np.savez_compressed(
            split_path,
            inputs=inputs,
            targets=targets,
            target_times_ns=target_times_ns,
            target_start_time_ns=target_start_time_ns,
            sample_index=sample_index,
            region_ids=np.asarray(selected_regions, dtype=np.int64),
        )
        outputs[split_name] = split_path.name
        split_meta[split_name] = {
            "path": str(split_path),
            "num_rows": int(frame.shape[0]),
            "num_samples": int(inputs.shape[0]),
            "time_start": str(frame.index.min()),
            "time_end": str(frame.index.max()),
        }
        if split_name == "train":
            train_target_scores = targets.mean(axis=(1, 2))

    if train_target_scores is None:
        raise RuntimeError("train split windows were not created")

    high_threshold = float(np.quantile(train_target_scores, _high_quantile(dataset_cfg)))
    low_threshold = float(np.quantile(train_target_scores, _low_quantile(dataset_cfg)))

    manifest = {
        "dataset_name": str(dataset_cfg.get("name", "telecom_italia_milan")),
        "dataset_version": str(dataset_cfg.get("dataset_version", "manual-local")),
        "source_urls": list(dataset_cfg.get("source_urls", [])),
        "raw_dir": str(_raw_dir(dataset_cfg)),
        "grid_path": str(dataset_cfg.get("grid_path", "")),
        "raw_file_count": 0 if raw_sources == ["synthetic_smoke_generator"] else len(raw_sources),
        "raw_schema": {
            "region_column": _region_column(dataset_cfg),
            "timestamp_column": _timestamp_column(dataset_cfg),
            "target_field": _target_column(dataset_cfg),
            "timestamp_unit": str(dataset_cfg.get("timestamp_unit", "ms")),
        },
        "aggregation": {
            "input_granularity": str(dataset_cfg.get("input_granularity", "10min")),
            "output_granularity": f"{int(dataset_cfg.get('aggregate_minutes', 60))}min",
            "bucket_rule": "floor_60min_utc",
            "value_aggregation": "sum",
        },
        "target_field": _target_column(dataset_cfg),
        "granularity": str(dataset_cfg.get("granularity", "1h")),
        "context_len": context_len,
        "horizon": horizon,
        "top_k_regions": len(selected_regions),
        "selected_regions": selected_regions,
        "selection": {
            "method": "top_k_by_train_period_sum",
            "selected_regions": selected_regions,
            "scores": selection_scores,
        },
        "normalization": {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        },
        "normalization_stats": {
            "method": "zscore_per_region_train_only",
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        },
        "thresholds": {
            "high_load": high_threshold,
            "low_load": low_threshold,
            "score_definition": "mean_over_horizon_and_regions_on_raw_scale",
        },
        "outputs": outputs,
        "splits": split_meta,
        "artifacts": outputs,
        "sources": raw_sources,
        "created_at": datetime.now(UTC).isoformat(),
        "notes": [
            "manual raw_dir used because auto-download is unreliable in the current environment",
        ],
    }

    manifest_path = Path(
        os.path.expandvars(str(dataset_cfg.get("manifest_path", prepared_dir / "manifest.json")))
    ).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="traffic_prediction_core/config/milan_prepare.yaml")
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--context-len", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_cfg = _dataset_cfg(config)
    if args.raw_dir:
        dataset_cfg["raw_dir"] = args.raw_dir
    if args.output_dir:
        dataset_cfg["output_dir"] = args.output_dir
        dataset_cfg["prepared_dir"] = args.output_dir
        dataset_cfg["manifest_path"] = str(Path(args.output_dir) / "manifest.json")
    if args.top_k > 0:
        dataset_cfg["top_k_regions"] = args.top_k
    if args.context_len > 0:
        dataset_cfg["context_len"] = args.context_len
    if args.horizon > 0:
        dataset_cfg["horizon"] = args.horizon
    if args.smoke:
        smoke_cfg = dataset_cfg.setdefault("smoke", {})
        smoke_cfg["enabled"] = True
    manifest = prepare_dataset(config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
