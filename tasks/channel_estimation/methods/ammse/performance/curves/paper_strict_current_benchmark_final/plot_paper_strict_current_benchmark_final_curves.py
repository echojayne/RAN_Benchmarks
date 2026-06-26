"""Plot retained paper-strict A-MMSE evaluation curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURVE_DIR = Path(__file__).resolve().parent
RAW_CSV = (
    CURVE_DIR.parents[1]
    / "raw_data"
    / "paper_strict_current_benchmark_final_eval"
    / "paper_strict_current_benchmark_final_metrics.csv"
)

AXES = {
    "snr": {
        "title": "SNR Robustness",
        "xlabel": "SNR (dB)",
        "output": "paper_strict_current_benchmark_final_mse_db_vs_snr.png",
    },
    "delay_spread": {
        "title": "Delay Spread Robustness",
        "xlabel": "Delay spread (ns)",
        "output": "paper_strict_current_benchmark_final_mse_db_vs_delay_spread.png",
    },
    "doppler": {
        "title": "Doppler Robustness",
        "xlabel": "Doppler (Hz)",
        "output": "paper_strict_current_benchmark_final_mse_db_vs_doppler.png",
    },
}


def load_rows() -> list[dict[str, str]]:
    with RAW_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rows_for_axis(rows: list[dict[str, str]], axis: str) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if row["axis"] == axis]
    selected.sort(key=lambda row: float(row["condition"]))
    return [float(row["condition"]) for row in selected], [float(row["mse_db"]) for row in selected]


def plot_single_axis(rows: list[dict[str, str]], axis: str) -> Path:
    xs, ys = rows_for_axis(rows, axis)
    cfg = AXES[axis]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#6b3f9b")
    ax.set_title(cfg["title"])
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("MSE (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output = CURVE_DIR / cfg["output"]
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def plot_combined(rows: list[dict[str, str]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for ax, axis in zip(axes, AXES, strict=True):
        xs, ys = rows_for_axis(rows, axis)
        cfg = AXES[axis]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color="#6b3f9b")
        ax.set_title(cfg["title"])
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel("MSE (dB)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output = CURVE_DIR / "paper_strict_current_benchmark_final_all_curves.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> int:
    rows = load_rows()
    outputs = [plot_single_axis(rows, axis) for axis in AXES]
    outputs.append(plot_combined(rows))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
