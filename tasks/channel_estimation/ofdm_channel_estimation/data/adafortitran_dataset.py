"""PyTorch dataset for official AdaFortiTran-format MATLAB data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from ofdm_channel_estimation.baselines.ls import ls_baseline_from_sparse_grid, ls_sparse_grid_from_mat
from ofdm_channel_estimation.data_gen.ofdm import OFDMConfig, build_pilot_mask
from ofdm_channel_estimation.eval.run_baselines import parse_metadata


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_split_dir(config: dict[str, Any], split: str) -> Path:
    dataset_cfg = config["dataset"]
    root_dir = Path(os.path.expandvars(str(dataset_cfg["root_dir"]))).expanduser()
    comparison_splits = {
        str(axis_name): str(rel_path)
        for axis_name, rel_path in dataset_cfg.get("comparison_splits", {}).items()
    }
    split_map = {
        "train": dataset_cfg["train_dir"],
        "val": dataset_cfg["val_dir"],
        "test": dataset_cfg["test_dir"],
        "snr": comparison_splits.get("snr", dataset_cfg["test_dir"]),
        "test_snr": comparison_splits.get("snr", dataset_cfg["test_dir"]),
        "delay_spread": comparison_splits.get("delay_spread", dataset_cfg["test_dir"]),
        "test_delay_spread": comparison_splits.get("delay_spread", dataset_cfg["test_dir"]),
        "doppler": comparison_splits.get("doppler", dataset_cfg["test_dir"]),
        "test_doppler": comparison_splits.get("doppler", dataset_cfg["test_dir"]),
    }
    if split not in split_map:
        raise ValueError(f"unsupported split: {split}")
    return root_dir / str(split_map[split])


def _analysis_axis_from_split(split: str) -> str:
    split_to_axis = {
        "train": "train",
        "val": "val",
        "test": "snr",
        "snr": "snr",
        "test_snr": "snr",
        "delay_spread": "delay_spread",
        "test_delay_spread": "delay_spread",
        "doppler": "doppler",
        "test_doppler": "doppler",
    }
    if split not in split_to_axis:
        raise ValueError(f"unsupported split: {split}")
    return split_to_axis[split]


def _relative_split_dir(config: dict[str, Any], split: str) -> str:
    dataset_cfg = config["dataset"]
    comparison_splits = {
        str(axis_name): str(rel_path)
        for axis_name, rel_path in dataset_cfg.get("comparison_splits", {}).items()
    }
    split_map = {
        "train": str(dataset_cfg["train_dir"]),
        "val": str(dataset_cfg["val_dir"]),
        "test": str(dataset_cfg["test_dir"]),
        "snr": comparison_splits.get("snr", str(dataset_cfg["test_dir"])),
        "test_snr": comparison_splits.get("snr", str(dataset_cfg["test_dir"])),
        "delay_spread": comparison_splits.get("delay_spread", str(dataset_cfg["test_dir"])),
        "test_delay_spread": comparison_splits.get("delay_spread", str(dataset_cfg["test_dir"])),
        "doppler": comparison_splits.get("doppler", str(dataset_cfg["test_dir"])),
        "test_doppler": comparison_splits.get("doppler", str(dataset_cfg["test_dir"])),
    }
    if split not in split_map:
        raise ValueError(f"unsupported split: {split}")
    return split_map[split]


def _pilot_mask_from_config(config: dict[str, Any]) -> np.ndarray:
    ofdm_cfg = config["ofdm"]
    pilot_cfg = config["pilot"]
    return build_pilot_mask(
        OFDMConfig(
            num_subcarriers=int(ofdm_cfg["num_subcarriers"]),
            num_symbols=int(ofdm_cfg["num_symbols"]),
            pilot_spacing_subcarriers=int(pilot_cfg["spacing_subcarriers"]),
            pilot_symbol_indices_zero_based=tuple(int(v) for v in pilot_cfg["symbol_indices_zero_based"]),
            pilot_offset_subcarriers=int(pilot_cfg["offset_subcarriers"]),
        )
    )


def _complex_to_channels(x: np.ndarray) -> np.ndarray:
    return np.stack((x.real, x.imag), axis=0).astype(np.float32)


def _official_pilot_vector_from_sparse_grid(h_ls_sparse: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Match the public AdaFortiTran MatDataset pilot extraction order.

    The official loader extracts non-zero pilot entries directly from the sparse
    LS grid and relies on PyTorch's row-major boolean indexing order. Our older
    loader used MATLAB-style Fortran flattening, which permuted pilot symbols and
    shifts the baseline away from the upstream implementation.
    """

    if h_ls_sparse.shape != pilot_mask.shape:
        raise ValueError("shape mismatch between sparse grid and pilot mask")
    pilot_values = np.asarray(h_ls_sparse[h_ls_sparse != 0], dtype=np.complex64)
    expected_pilots = int(pilot_mask.sum())
    if pilot_values.size != expected_pilots:
        raise ValueError(f"expected {expected_pilots} pilot values, got {pilot_values.size}")
    return pilot_values


def mat_to_sample(
    h: np.ndarray,
    *,
    pilot_mask: np.ndarray,
    normalize_inputs: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert official `H` data to model-ready tensors.

    Returned tensors expose both:
    - paper-minimal ingredients: sparse LS grid, pilot mask, pilot vector
    - mainline contract helpers: LS coarse estimate and concatenated inputs
    """

    h_true = np.asarray(h[:, :, 0], dtype=np.complex64)
    h_ls_sparse = np.asarray(ls_sparse_grid_from_mat(h), dtype=np.complex64)
    h_ls_coarse = np.asarray(ls_baseline_from_sparse_grid(h_ls_sparse), dtype=np.complex64)
    h_pilot_vector = _official_pilot_vector_from_sparse_grid(h_ls_sparse, pilot_mask)

    scale = np.sqrt(np.mean(np.abs(h_ls_coarse) ** 2)).astype(np.float32)
    if not np.isfinite(scale) or float(scale) <= 1e-8:
        scale = np.float32(1.0)

    if normalize_inputs:
        h_true = h_true / scale
        h_ls_sparse = h_ls_sparse / scale
        h_ls_coarse = h_ls_coarse / scale
        h_pilot_vector = h_pilot_vector / scale

    sparse_grid = torch.from_numpy(_complex_to_channels(h_ls_sparse))
    ls_coarse_grid = torch.from_numpy(_complex_to_channels(h_ls_coarse))
    pilot_vector = torch.from_numpy(_complex_to_channels(h_pilot_vector))
    target = torch.from_numpy(_complex_to_channels(h_true))
    pilot_mask_tensor = torch.from_numpy(pilot_mask.astype(np.bool_, copy=False))
    inputs = torch.cat((sparse_grid, ls_coarse_grid), dim=0)

    return {
        "inputs": inputs,
        "sparse_ls_grid": sparse_grid,
        "ls_coarse_grid": ls_coarse_grid,
        "pilot_mask": pilot_mask_tensor,
        "pilot_vector": pilot_vector,
        "target": target,
        "scale": torch.tensor(float(scale), dtype=torch.float32),
    }


class AdaFortiTranMatDataset(Dataset[dict[str, Any]]):
    """Expose sparse/dense LS tensors and pilot vectors for training."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        *,
        split: str,
        limit: int = 0,
        normalize_inputs: bool = True,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config
        self.split = split
        self.normalize_inputs = normalize_inputs
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
        metadata["analysis_axis"] = self.analysis_axis
        metadata["analysis_split_dir"] = self.relative_split_dir

        sample = mat_to_sample(
            h,
            pilot_mask=self.pilot_mask,
            normalize_inputs=self.normalize_inputs,
        )
        sample["noise_var"] = torch.tensor(float(np.asarray(data.get("var_hat", np.nan)).squeeze()), dtype=torch.float32)
        sample["metadata"] = metadata
        return sample


def build_split_dataset(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    limit: int = 0,
    normalize_inputs: bool = True,
) -> AdaFortiTranMatDataset:
    return AdaFortiTranMatDataset(
        config,
        split=split,
        limit=limit,
        normalize_inputs=normalize_inputs,
    )


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_fields = (
        "inputs",
        "sparse_ls_grid",
        "ls_coarse_grid",
        "pilot_vector",
        "target",
        "scale",
        "noise_var",
    )
    collated: dict[str, Any] = {name: torch.stack([item[name] for item in batch], dim=0) for name in tensor_fields}
    collated["pilot_mask"] = batch[0]["pilot_mask"]
    collated["metadata"] = [item["metadata"] for item in batch]
    return collated


def build_split_dataloader(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    limit: int = 0,
    normalize_inputs: bool = True,
    sampler: Sampler[Any] | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    dataset = build_split_dataset(
        config,
        split=split,
        limit=limit,
        normalize_inputs=normalize_inputs,
    )
    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_batch,
    )


def build_split_loader(
    config: dict[str, Any] | str | Path,
    *,
    split: str,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    limit: int = 0,
    normalize_inputs: bool = True,
    sampler: Sampler[Any] | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Backward-compatible alias for older local imports."""

    return build_split_dataloader(
        config,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        limit=limit,
        normalize_inputs=normalize_inputs,
        sampler=sampler,
        pin_memory=pin_memory,
    )
