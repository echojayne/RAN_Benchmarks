"""Dataset validation and manifest generation for official external data."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import scipy.io as sio
import yaml

from ofdm_channel_estimation.data_gen.channel import ADAFORTITRAN_REFERENCE, AMMSE_REFERENCE
from ofdm_channel_estimation.data_gen.manifest import write_manifest


ADAFORTITRAN_FILE_RE = re.compile(
    r"^(?P<index>\d+)_SNR-(?P<snr>-?\d+)_DS-(?P<ds>\d+)_DOP-(?P<dop>\d+)_N-(?P<n>\d+)_(?P<profile>[A-Z\-]+)\.mat$"
)


class ExternalDataRequiredError(RuntimeError):
    """Raised when official data is required but not locally available."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _summarize_mat(path: Path, expected_shape: tuple[int, int]) -> dict[str, Any]:
    data = sio.loadmat(path)
    if "H" not in data:
        raise ValueError(f"{path} is missing key 'H'")
    h = data["H"]
    if h.shape != (*expected_shape, 2):
        raise ValueError(f"{path} has H shape {h.shape}, expected {expected_shape + (2,)}")
    summary = {
        "path": str(path),
        "h_shape": list(h.shape),
    }
    if "var_hat" in data:
        try:
            summary["var_hat"] = float(data["var_hat"].squeeze())
        except Exception:
            summary["var_hat"] = None
    return summary


def summarize_adafortitran_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    ofdm_cfg = config["ofdm"]
    root_dir = Path(os.path.expandvars(str(dataset_cfg["root_dir"]))).expanduser()
    if not root_dir.exists():
        raise ExternalDataRequiredError(
            "AdaFortiTran official dataset root does not exist. "
            "Generate it with OFDMChannelGenerator or provide the dataset directory."
        )

    expected_shape = (int(ofdm_cfg["num_subcarriers"]), int(ofdm_cfg["num_symbols"]))
    split_paths = {
        "train": root_dir / dataset_cfg["train_dir"],
        "val": root_dir / dataset_cfg["val_dir"],
        "test": root_dir / dataset_cfg["test_dir"],
    }

    manifest: dict[str, Any] = {
        "reference": ADAFORTITRAN_REFERENCE.__dict__,
        "dataset_root": str(root_dir),
        "expected_h_shape": [*expected_shape, 2],
        "splits": {},
    }

    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise ExternalDataRequiredError(f"Required split directory is missing: {split_path}")
        mat_files = sorted(split_path.rglob("*.mat"))
        if not mat_files:
            raise ExternalDataRequiredError(f"No .mat files found under {split_path}")

        first_file = mat_files[0]
        file_summary = _summarize_mat(first_file, expected_shape)
        split_summary: dict[str, Any] = {
            "path": str(split_path),
            "count": len(mat_files),
            "example": file_summary,
        }
        if split_name in {"train", "val"}:
            match = ADAFORTITRAN_FILE_RE.match(first_file.name)
            if match:
                split_summary["example_metadata"] = match.groupdict()
        manifest["splits"][split_name] = split_summary

    return manifest


def ammse_help_manifest(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    return {
        "reference": AMMSE_REFERENCE.__dict__,
        "required_files": {
            "perfect_mat_path": dataset_cfg["perfect_mat_path"],
            "noisy_mat_path": dataset_cfg["noisy_mat_path"],
        },
        "status": "upstream_repo_expects_external_mat_files",
    }


def build_manifest(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    fmt = config["dataset"]["format"]
    if fmt == "adafortitran_mat_dir":
        return summarize_adafortitran_dataset(config)
    if fmt == "ammse_pair_mat":
        return ammse_help_manifest(config)
    raise ValueError(f"Unsupported dataset format: {fmt}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest for official channel-estimation reference data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", required=True, help="Path to output manifest JSON.")
    args = parser.parse_args()

    manifest = build_manifest(args.config)
    write_manifest(args.output, manifest)
    print(f"Wrote manifest to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
