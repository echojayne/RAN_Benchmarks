"""OFDM and pilot-grid helpers aligned with the public AdaFortiTran generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OFDMConfig:
    num_subcarriers: int
    num_symbols: int
    pilot_spacing_subcarriers: int
    pilot_symbol_indices_zero_based: tuple[int, ...]
    pilot_offset_subcarriers: int | None = None


def compute_symmetric_offset(spacing: int, num_subcarriers: int) -> int:
    """Return the symmetric pilot offset described in OFDMChannelGenerator.

    Falls back to zero when perfect symmetry is not achievable.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    rhs = (num_subcarriers - 1) % spacing
    for offset in range(spacing):
        if (2 * offset) % spacing == rhs:
            return offset
    return 0


def build_pilot_mask(config: OFDMConfig) -> np.ndarray:
    """Build a dense [num_subcarriers, num_symbols] boolean pilot mask."""
    offset = config.pilot_offset_subcarriers
    if offset is None:
        offset = compute_symmetric_offset(
            config.pilot_spacing_subcarriers,
            config.num_subcarriers,
        )
    pilot_rows = np.arange(offset, config.num_subcarriers, config.pilot_spacing_subcarriers)
    mask = np.zeros((config.num_subcarriers, config.num_symbols), dtype=bool)
    mask[np.ix_(pilot_rows, np.asarray(config.pilot_symbol_indices_zero_based, dtype=int))] = True
    return mask


def flatten_fortran(x: np.ndarray) -> np.ndarray:
    """Match MATLAB's column-major flattening order."""
    return np.asarray(x).reshape(-1, order="F")


def sparse_pilot_vector(h_sparse: np.ndarray, pilot_mask: np.ndarray) -> np.ndarray:
    """Extract pilot observations in the same order as MATLAB logical indexing."""
    if h_sparse.shape != pilot_mask.shape:
        raise ValueError("shape mismatch between sparse grid and pilot mask")
    return flatten_fortran(h_sparse)[flatten_fortran(pilot_mask)]


def _interp_complex_1d(values: np.ndarray, known_idx: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    if known_idx.size < 2:
        raise ValueError("at least two known points are required for linear interpolation with extrapolation")

    known_values = values[known_idx]
    real = np.interp(target_idx, known_idx, known_values.real)
    imag = np.interp(target_idx, known_idx, known_values.imag)

    left_mask = target_idx < known_idx[0]
    if np.any(left_mask):
        left_span = float(known_idx[1] - known_idx[0])
        left_slope = (known_values[1] - known_values[0]) / left_span
        left_delta = target_idx[left_mask] - known_idx[0]
        extrapolated = known_values[0] + left_slope * left_delta
        real[left_mask] = extrapolated.real
        imag[left_mask] = extrapolated.imag

    right_mask = target_idx > known_idx[-1]
    if np.any(right_mask):
        right_span = float(known_idx[-1] - known_idx[-2])
        right_slope = (known_values[-1] - known_values[-2]) / right_span
        right_delta = target_idx[right_mask] - known_idx[-1]
        extrapolated = known_values[-1] + right_slope * right_delta
        real[right_mask] = extrapolated.real
        imag[right_mask] = extrapolated.imag

    return real + 1j * imag


def bilinear_interp(h_ls_sparse: np.ndarray) -> np.ndarray:
    """Replicate the row-then-column interpolation in bilinear_interp.m."""
    h_hat = np.array(h_ls_sparse, dtype=np.complex128, copy=True)
    n_row, n_col = h_hat.shape

    for row in range(n_row):
        row_values = h_hat[row, :]
        known_idx = np.flatnonzero(row_values != 0)
        if known_idx.size >= 2:
            h_hat[row, :] = _interp_complex_1d(row_values, known_idx, np.arange(n_col))

    for col in range(n_col):
        col_values = h_hat[:, col]
        known_idx = np.flatnonzero(col_values != 0)
        if known_idx.size >= 2:
            h_hat[:, col] = _interp_complex_1d(col_values, known_idx, np.arange(n_row))

    return h_hat
