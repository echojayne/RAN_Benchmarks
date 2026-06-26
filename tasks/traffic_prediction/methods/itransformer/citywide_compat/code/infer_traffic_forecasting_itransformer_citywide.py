from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from itransformer_citywide import (  # noqa: E402
    CitywideITransformer,
    build_time_features,
    compute_per_cell_scaler,
    load_excluded_cell_ids,
    make_batch,
    prepare_hourly_matrix,
    resolve_time_splits,
)
from run_traffic_forecasting_itransformer_citywide import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "config" / "itransformer_citywide.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(Path(os.path.expandvars(args.config)).expanduser())
    output_dir = Path(config.output_dir)
    checkpoint_path = (
        Path(os.path.expandvars(args.checkpoint)).expanduser()
        if args.checkpoint
        else (output_dir / "checkpoints" / "best.pt")
    )

    matrix, timestamps, cell_ids = prepare_hourly_matrix(Path(config.raw_data_dir), Path(config.cache_dir))
    excluded = load_excluded_cell_ids(config.exclude_cell_ids_path)
    if excluded.size > 0:
        keep_mask = ~np.isin(cell_ids, excluded)
        matrix = matrix[:, keep_mask]
        cell_ids = cell_ids[keep_mask]

    time_features = build_time_features(timestamps)
    split = resolve_time_splits(config, timestamps)
    train_start_idx = split["train_start_idx"]
    val_start_idx = split["val_start_idx"]
    test_start_idx = split["test_start_idx"]
    test_end_idx = split["test_end_idx"]
    mins, scales = compute_per_cell_scaler(matrix[train_start_idx:val_start_idx])
    norm_matrix = ((matrix - mins[None, :]) / scales[None, :]).astype(np.float32)
    target_times = np.arange(test_start_idx, test_end_idx, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CitywideITransformer(
        context_len=config.context_len,
        num_variates=1 + time_features.shape[-1],
        num_cells=matrix.shape[1],
        d_model=config.d_model,
        depth=config.depth,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        cell_embedding_dim=config.cell_embedding_dim,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    preds = np.zeros((len(target_times), matrix.shape[1]), dtype=np.float32)
    with torch.no_grad():
        all_cells = np.arange(matrix.shape[1], dtype=np.int64)
        for row_idx, target_time in enumerate(target_times):
            x_np, _ = make_batch(norm_matrix, time_features, int(target_time), all_cells, config.context_len)
            x = torch.from_numpy(x_np).to(device)
            cell_tensor = torch.from_numpy(all_cells).to(device)
            preds[row_idx] = model(x, cell_tensor).detach().cpu().numpy().astype(np.float32)

    preds_raw = preds * scales[None, :] + mins[None, :]
    trues_raw = matrix[test_start_idx:test_end_idx]
    save_path = (
        Path(os.path.expandvars(args.output)).expanduser()
        if args.output
        else (output_dir / "predictions_test.npz")
    )
    np.savez_compressed(
        save_path,
        predictions=preds_raw,
        targets=trues_raw,
        cell_ids=cell_ids,
        timestamps=np.asarray([ts.isoformat() for ts in timestamps[target_times]], dtype=object),
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "output": str(save_path),
        "num_cells": int(matrix.shape[1]),
        "num_steps": int(len(target_times)),
        "test_start": timestamps[target_times[0]].isoformat(),
        "test_end": timestamps[target_times[-1]].isoformat(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
