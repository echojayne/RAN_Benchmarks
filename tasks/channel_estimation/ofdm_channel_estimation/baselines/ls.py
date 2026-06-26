"""Least-squares helpers aligned with the public AdaFortiTran data contract."""

from __future__ import annotations

import numpy as np

from ofdm_channel_estimation.data_gen.ofdm import bilinear_interp


def ls_sparse_grid_from_mat(h: np.ndarray) -> np.ndarray:
    """Extract the sparse LS grid from the official `H[:,:,1]` slot."""

    if h.ndim != 3 or h.shape[2] < 2:
        raise ValueError(f"expected H with shape [subcarriers, symbols, >=2], got {h.shape}")
    return np.asarray(h[:, :, 1], dtype=np.complex128)


def ls_baseline_from_sparse_grid(h_ls_sparse: np.ndarray) -> np.ndarray:
    """Reconstruct a dense LS estimate with the public bilinear interpolation rule."""

    return np.asarray(bilinear_interp(h_ls_sparse), dtype=np.complex128)
