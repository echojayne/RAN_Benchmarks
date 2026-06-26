"""Prepared Milan dataset loader."""

from __future__ import annotations

import yaml
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_prepared_dir(config: dict[str, Any]) -> Path:
    data_cfg = config.get("data", config.get("dataset", {}))
    prepared_dir = data_cfg.get("prepared_dir")
    if prepared_dir is None:
        raise ValueError("config must define data.prepared_dir")
    return Path(os.path.expandvars(str(prepared_dir))).expanduser()


def load_manifest(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_or_path, (str, Path)):
        config = load_yaml_config(config_or_path)
    else:
        config = config_or_path
    manifest_path = _resolve_prepared_dir(config) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_split_npz(prepared_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = prepared_dir / f"{split}.npz"
    if not path.exists():
        raise ValueError(f"prepared split file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _resolve_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        return load_yaml_config(config)
    return config


class MilanPreparedDataset(Dataset[dict[str, Any]]):
    """Expose prepared forecasting windows."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        *,
        split: str,
        limit: int = 0,
        normalize: bool = True,
    ) -> None:
        self.config = _resolve_config(config)
        self.normalize = normalize
        self.prepared_dir = _resolve_prepared_dir(self.config)
        self.manifest = load_manifest(self.config)
        self.split = split
        arrays = _load_split_npz(self.prepared_dir, split)
        self.inputs = arrays["inputs"].astype(np.float32)
        self.targets = arrays["targets"].astype(np.float32)
        self.target_times_ns = arrays["target_times_ns"].astype(np.int64)
        self.target_start_time_ns = arrays["target_start_time_ns"].astype(np.int64)
        self.sample_index = arrays["sample_index"].astype(np.int64)
        if limit > 0:
            self.inputs = self.inputs[:limit]
            self.targets = self.targets[:limit]
            self.target_times_ns = self.target_times_ns[:limit]
            self.target_start_time_ns = self.target_start_time_ns[:limit]
            self.sample_index = self.sample_index[:limit]

        stats = self.manifest["normalization"]
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        self.selected_regions = list(self.manifest.get("selected_regions", []))
        self.normalization_stats = dict(self.manifest.get("normalization_stats", self.manifest.get("normalization", {})))

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_inputs = self.inputs[index]
        raw_targets = self.targets[index]
        if self.normalize:
            inputs = (raw_inputs - self.mean) / self.std
            targets = (raw_targets - self.mean) / self.std
        else:
            inputs = raw_inputs
            targets = raw_targets
        return {
            "inputs": torch.from_numpy(inputs.astype(np.float32, copy=False)),
            "targets": torch.from_numpy(targets.astype(np.float32, copy=False)),
            "target_times_ns": torch.from_numpy(self.target_times_ns[index]),
            "target_start_time_ns": torch.tensor(int(self.target_start_time_ns[index]), dtype=torch.int64),
            "sample_index": torch.tensor(int(self.sample_index[index]), dtype=torch.int64),
            "metadata": {
                "split": self.split,
                "sample_index": int(self.sample_index[index]),
                "target_start_time_ns": int(self.target_start_time_ns[index]),
                "target_times_ns": self.target_times_ns[index].astype(np.int64, copy=False).tolist(),
                "selected_regions": self.selected_regions,
                "normalization_stats": self.normalization_stats,
            },
        }


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "target_times_ns": torch.stack([item["target_times_ns"] for item in batch], dim=0),
        "target_start_time_ns": torch.stack([item["target_start_time_ns"] for item in batch], dim=0),
        "sample_index": torch.stack([item["sample_index"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
    }


def build_split_dataset(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    limit: int = 0,
    normalize: bool = True,
) -> MilanPreparedDataset:
    return MilanPreparedDataset(config, split=split, limit=limit, normalize=normalize)


def build_split_dataloader(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    limit: int = 0,
    normalize: bool = True,
) -> DataLoader:
    dataset = build_split_dataset(config, split=split, limit=limit, normalize=normalize)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


load_yaml = load_yaml_config
