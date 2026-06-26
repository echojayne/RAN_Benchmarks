"""PyTorch dataset for A-MMSE on AdaFortiTran-format MATLAB data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset

from ofdm_channel_estimation.data.adafortitran_dataset import (
    _analysis_axis_from_split,
    _pilot_mask_from_config,
    _relative_split_dir,
    _resolve_split_dir,
    load_config,
)
from ofdm_channel_estimation.data_gen.ofdm import sparse_pilot_vector
from ofdm_channel_estimation.eval.run_baselines import parse_metadata


def _complex_to_channels(x: np.ndarray) -> np.ndarray:
    return np.stack((x.real, x.imag), axis=0).astype(np.float32)


def mat_to_ammse_sample(
    h: np.ndarray,
    *,
    pilot_mask: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Convert official `H` data to the minimal A-MMSE tensor contract."""

    h_true = np.asarray(h[:, :, 0], dtype=np.complex64)
    h_pilot_sparse = np.asarray(h[:, :, 1], dtype=np.complex64)
    h_pilot_vector = np.asarray(sparse_pilot_vector(h_pilot_sparse, pilot_mask), dtype=np.complex64)

    return {
        "pilot_vector": torch.from_numpy(_complex_to_channels(h_pilot_vector)),
        "pilot_mask": torch.from_numpy(pilot_mask.astype(np.bool_, copy=False)),
        "target_full_grid": torch.from_numpy(_complex_to_channels(h_true)),
    }


class AMMSEMatDataset(Dataset[dict[str, Any]]):
    """Expose pilot-domain observations and dense targets for A-MMSE."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        *,
        split: str,
        limit: int = 0,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config
        self.split = split
        self.pilot_mask = _pilot_mask_from_config(config)
        split_dir = _resolve_split_dir(config, split)
        self.analysis_axis = _analysis_axis_from_split(split)
        self.relative_split_dir = _relative_split_dir(config, split)
        self.files = sorted(path for path in split_dir.rglob("*.mat") if path.is_file())
        if limit > 0:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError(f"no .mat files found for split {split} under {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        data = sio.loadmat(path)
        h = np.asarray(data["H"], dtype=np.complex128)
        metadata = parse_metadata(path.name)
        metadata["path"] = str(path)
        metadata["pilot_count"] = int(self.pilot_mask.sum())
        metadata["full_grid_shape"] = [int(v) for v in h.shape[:2]]
        metadata["analysis_axis"] = self.analysis_axis
        metadata["analysis_split_dir"] = self.relative_split_dir

        sample = mat_to_ammse_sample(h, pilot_mask=self.pilot_mask)
        sample["noise_var"] = torch.tensor(
            float(np.asarray(data.get("var_hat", np.nan)).squeeze()),
            dtype=torch.float32,
        )
        sample["metadata"] = metadata
        return sample


def build_ammse_dataset(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    limit: int = 0,
) -> AMMSEMatDataset:
    return AMMSEMatDataset(
        config,
        split=split,
        limit=limit,
    )


def _collate_ammse_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "pilot_vector": torch.stack([item["pilot_vector"] for item in batch], dim=0),
        "target_full_grid": torch.stack([item["target_full_grid"] for item in batch], dim=0),
        "noise_var": torch.stack([item["noise_var"] for item in batch], dim=0),
    }
    collated["pilot_mask"] = batch[0]["pilot_mask"]
    collated["metadata"] = [item["metadata"] for item in batch]
    return collated


def build_ammse_dataloader(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    limit: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    dataset = build_ammse_dataset(
        config,
        split=split,
        limit=limit,
    )
    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_ammse_batch,
    )
