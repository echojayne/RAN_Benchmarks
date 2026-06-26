"""Run an ML-TP published-baseline week sweep on the current paper-aligned benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic_prediction_core.data.prepare_milan import load_yaml
from traffic_prediction_core.paper.mltp import (
    MLTPSample,
    PaperMLTPLSTMRegressor,
    build_weekly_meta_feature,
    fit_mltp_base_learner,
    make_fiwv,
    select_mltp_initial_state,
)
from traffic_prediction_core.paper.run_cell_benchmark import (
    RegimeSpec,
    _build_filtered_hourly_pivot,
    _generate_windows,
    _load_candidate_cells,
    _normalize_cell,
    _select_cells,
    _split_train_val,
    _timestamp,
)
from traffic_prediction_core.train.common import resolve_device, set_seed


def _generate_all_windows(
    cell_series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    context_len: int,
    end_time: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    loads = cell_series.astype(np.float32, copy=False)
    dow = (timestamps.dayofweek.to_numpy(dtype=np.float32) / 6.0).astype(np.float32, copy=False)
    hod = (timestamps.hour.to_numpy(dtype=np.float32) / 23.0).astype(np.float32, copy=False)
    features = np.stack([loads, dow, hod], axis=1)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for end in range(context_len - 1, len(timestamps) - 1):
        target_index = end + 1
        target_time = timestamps[target_index]
        if target_time >= end_time:
            continue
        inputs.append(features[end - context_len + 1 : end + 1])
        targets.append(features[target_index, 0])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-path", default="outputs/traffic_prediction/paper_aligned/filtered_hourly_cache.npz")
    parser.add_argument("--weeks", type=int, default=5)
    parser.add_argument("--cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--existing-week-sweep-csv")
    return parser.parse_args()


def _build_week_regimes(config: dict[str, Any], max_weeks: int) -> list[RegimeSpec]:
    bench_cfg = config["benchmark"]
    start = _timestamp(str(bench_cfg["fine_tune_start"]))
    test_start = _timestamp(str(bench_cfg["test_start"]))
    test_end = _timestamp(str(bench_cfg["test_end"]))
    regimes = []
    for week in range(1, max_weeks + 1):
        regimes.append(
            RegimeSpec(
                name=f"week_{week}",
                train_start=start,
                train_end=start + pd.Timedelta(days=7 * week),
                test_start=test_start,
                test_end=test_end,
            )
        )
    return regimes


def _to_db(values: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return 20.0 * np.log10(clipped)


def _plot_week_sweep(df: pd.DataFrame, *, output_dir: Path, db_scale: bool) -> tuple[Path, Path]:
    styles = {
        "naive": {"label": "Naive", "marker": "s", "color": "#2ca02c"},
        "lstm": {"label": "LSTM", "marker": "^", "color": "#ff7f0e"},
        "fixed_itransformer": {"label": "Fixed iTransformer", "marker": "D", "color": "#d62728"},
        "mltp": {"label": "ML-TP (published)", "marker": "P", "color": "#9467bd"},
    }
    fig, ax = plt.subplots(figsize=(8.9, 5.5), constrained_layout=True)
    y_col = "mean_MAE_dB" if db_scale else "mean_MAE"
    lower_col = "lower_MAE_dB" if db_scale else "lower_MAE"
    upper_col = "upper_MAE_dB" if db_scale else "upper_MAE"
    for model_name, style in styles.items():
        subset = df[df["model"] == model_name].sort_values("weeks")
        if subset.empty:
            continue
        ax.plot(subset["weeks"], subset[y_col], marker=style["marker"], color=style["color"], linewidth=1.8, label=style["label"])
        ax.fill_between(subset["weeks"], subset[lower_col], subset[upper_col], color=style["color"], alpha=0.12, linewidth=0.0)
    ax.set_xlabel("Training Window (weeks)")
    ax.set_ylabel("Mean MAE (dB)" if db_scale else "Mean MAE")
    ax.set_xticks(sorted(df["weeks"].unique().tolist()))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="best")
    suffix = "_db" if db_scale else ""
    png_path = output_dir / f"paper_aligned_week_sweep_with_mltp{suffix}.png"
    pdf_path = output_dir / f"paper_aligned_week_sweep_with_mltp{suffix}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.baseline_config)
    if args.cells > 0:
        cfg.setdefault("benchmark", {})["cells_override"] = args.cells
    if args.seed > 0:
        cfg.setdefault("training", {})["seed"] = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["training"].get("seed", 17))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = resolve_device(str(cfg["training"].get("device", "auto")))

    base_prepare_cfg = load_yaml(cfg["source_prepare_config"])
    include_cells = [int(value) for value in cfg["benchmark"].get("include_cells", [])]
    candidate_cells = sorted(set(_load_candidate_cells(cfg)) | set(include_cells))
    pivot, raw_sources = _build_filtered_hourly_pivot(
        base_prepare_cfg["dataset"],
        candidate_cells=candidate_cells,
        cache_path=Path(args.cache_path),
    )

    bench_cfg = cfg["benchmark"]
    start_time = _timestamp(str(bench_cfg["series_start"]))
    series_end = _timestamp(str(bench_cfg["series_end"]))
    pivot = pivot.loc[(pivot.index >= start_time) & (pivot.index < series_end)].copy()
    selected_cells = _select_cells(pivot, cfg)
    candidate_pool = [int(value) for value in pivot.columns.astype(int).tolist()]
    regimes = _build_week_regimes(cfg, args.weeks)

    training_cfg = cfg["training"]
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    batch_size = int(training_cfg.get("batch_size", 64))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    epochs = int(training_cfg.get("epochs", 30))
    patience = int(training_cfg.get("patience", 5))
    val_fraction = float(training_cfg.get("val_fraction", 0.1))

    fiwv = make_fiwv(PaperMLTPLSTMRegressor())

    aggregate_rows: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    meta_feature_start = _timestamp(str(bench_cfg["fine_tune_start"]))
    meta_feature_end = min(meta_feature_start + pd.Timedelta(hours=168), _timestamp(str(bench_cfg["test_end"])))
    meta_feature_mask = np.asarray((pivot.index >= meta_feature_start) & (pivot.index < meta_feature_end), dtype=bool)

    donor_pool_data: dict[int, dict[str, Any]] = {}
    for cell_id in candidate_pool:
        normalized = _normalize_cell(
            pivot[cell_id],
            start_time=_timestamp(str(bench_cfg["normalization_start"])),
            end_time=_timestamp(str(bench_cfg["normalization_end"])),
        )
        if normalized is None:
            continue
        full_inputs, full_targets = _generate_all_windows(
            normalized,
            pivot.index,
            context_len=int(bench_cfg.get("context_len", 3)),
            end_time=_timestamp(str(bench_cfg["test_end"])),
        )
        if len(full_inputs) < 20:
            continue
        train_inputs, train_targets, val_inputs, val_targets = _split_train_val(
            full_inputs,
            full_targets,
            val_fraction=val_fraction,
        )
        donor_pool_data[int(cell_id)] = {
            "meta_feature": build_weekly_meta_feature(normalized[meta_feature_mask]),
            "train_inputs": train_inputs,
            "train_targets": train_targets,
            "val_inputs": val_inputs,
            "val_targets": val_targets,
        }

    meta_samples: list[MLTPSample] = []
    meta_bank_rows: list[dict[str, Any]] = []
    for cell_id, data in donor_pool_data.items():
        model, train_meta = fit_mltp_base_learner(
            data["train_inputs"],
            data["train_targets"],
            data["val_inputs"],
            data["val_targets"],
            device=device,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            initial_state=fiwv,
        )
        meta_samples.append(
            MLTPSample(
                cell_id=int(cell_id),
                meta_feature=data["meta_feature"],
                state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            )
        )
        meta_bank_rows.append({"cell_id": int(cell_id), "best_epoch": int(train_meta["best_epoch"]), "best_val_mae": float(train_meta["best_val_mae"])})
    pd.DataFrame(meta_bank_rows).to_csv(output_dir / "mltp_meta_bank.csv", index=False)

    for regime in regimes:
        cell_data: dict[int, dict[str, Any]] = {}
        for cell_id in selected_cells:
            normalized = _normalize_cell(
                pivot[cell_id],
                start_time=_timestamp(str(bench_cfg["normalization_start"])),
                end_time=_timestamp(str(bench_cfg["normalization_end"])),
            )
            if normalized is None:
                continue
            fine_inputs, fine_targets, test_inputs, test_targets = _generate_windows(
                normalized,
                pivot.index,
                context_len=int(bench_cfg.get("context_len", 3)),
                regime=regime,
            )
            train_inputs, train_targets, val_inputs, val_targets = _split_train_val(
                fine_inputs,
                fine_targets,
                val_fraction=val_fraction,
            )
            cell_data[int(cell_id)] = {
                "meta_feature": donor_pool_data[int(cell_id)]["meta_feature"],
                "fine_inputs": fine_inputs,
                "fine_targets": fine_targets,
                "train_inputs": train_inputs,
                "train_targets": train_targets,
                "val_inputs": val_inputs,
                "val_targets": val_targets,
                "test_inputs": test_inputs,
                "test_targets": test_targets,
            }

        model_maes: list[float] = []
        model_rmses: list[float] = []
        model_mses: list[float] = []
        for cell_id, data in cell_data.items():
            donor_sample, donor_candidates = select_mltp_initial_state(
                data["meta_feature"],
                [sample for sample in meta_samples if sample.cell_id != int(cell_id)],
                target_finetune_inputs=data["fine_inputs"],
                target_finetune_targets=data["fine_targets"],
                device=device,
                eval_batch_size=eval_batch_size,
                k_neighbors=int(args.k_neighbors),
            )
            for row in donor_candidates:
                candidate_rows.append({"regime": regime.name, "target_cell_id": int(cell_id), "selected_donor_cell_id": int(donor_sample.cell_id), **row})
            model, train_meta = fit_mltp_base_learner(
                data["train_inputs"],
                data["train_targets"],
                data["val_inputs"],
                data["val_targets"],
                device=device,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                initial_state=donor_sample.state_dict,
            )
            from traffic_prediction_core.paper.run_cell_benchmark import _evaluate_model

            eval_result = _evaluate_model(
                model,
                data["test_inputs"],
                data["test_targets"],
                device=device,
                batch_size=eval_batch_size,
            )
            model_maes.append(float(eval_result["MAE"]))
            model_rmses.append(float(eval_result["RMSE"]))
            model_mses.append(float(eval_result["MSE"]))
            per_cell_rows.append(
                {
                    "regime": regime.name,
                    "weeks": int(regime.name.split("_")[-1]),
                    "model": "mltp",
                    "cell_id": int(cell_id),
                    "donor_cell_id": int(donor_sample.cell_id),
                    "best_val_mae": float(train_meta["best_val_mae"]),
                    "best_epoch": int(train_meta["best_epoch"]),
                    "test_MAE": float(eval_result["MAE"]),
                    "test_RMSE": float(eval_result["RMSE"]),
                    "test_MSE": float(eval_result["MSE"]),
                }
            )

        aggregate_rows.append(
            {
                "regime": regime.name,
                "weeks": int(regime.name.split("_")[-1]),
                "model": "mltp",
                "num_cells": len(model_maes),
                "mean_MAE": float(np.mean(model_maes)),
                "std_MAE": float(np.std(model_maes)),
                "median_MAE": float(np.median(model_maes)),
                "mean_RMSE": float(np.mean(model_rmses)),
                "mean_MSE": float(np.mean(model_mses)),
            }
        )

    mltp_df = pd.DataFrame(aggregate_rows).sort_values("weeks")
    per_cell_df = pd.DataFrame(per_cell_rows).sort_values(["weeks", "cell_id"])
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["regime", "target_cell_id", "distance", "initial_MAE"])
    mltp_csv = output_dir / "week_sweep_mltp_metrics.csv"
    per_cell_csv = output_dir / "week_sweep_mltp_per_cell.csv"
    candidate_csv = output_dir / "week_sweep_mltp_candidates.csv"
    mltp_df.to_csv(mltp_csv, index=False)
    per_cell_df.to_csv(per_cell_csv, index=False)
    candidate_df.to_csv(candidate_csv, index=False)

    merged_df = mltp_df.copy()
    merged_csv = None
    merged_csv_db = None
    plot_png = None
    plot_pdf = None
    plot_db_png = None
    plot_db_pdf = None
    if args.existing_week_sweep_csv:
        existing_df = pd.read_csv(args.existing_week_sweep_csv)
        merged_df = pd.concat([existing_df, mltp_df], ignore_index=True, sort=False).sort_values(["weeks", "model"])
        merged_df["lower_MAE"] = np.clip(merged_df["mean_MAE"] - merged_df["std_MAE"], 1e-12, None)
        merged_df["upper_MAE"] = merged_df["mean_MAE"] + merged_df["std_MAE"]
        merged_df["mean_MAE_dB"] = _to_db(merged_df["mean_MAE"])
        merged_df["lower_MAE_dB"] = _to_db(merged_df["lower_MAE"])
        merged_df["upper_MAE_dB"] = _to_db(merged_df["upper_MAE"])
        merged_csv = output_dir / "week_sweep_aggregate_with_mltp.csv"
        merged_csv_db = output_dir / "week_sweep_aggregate_with_mltp_db.csv"
        merged_df.drop(columns=["mean_MAE_dB", "lower_MAE_dB", "upper_MAE_dB"]).to_csv(merged_csv, index=False)
        merged_df.to_csv(merged_csv_db, index=False)
        plot_png, plot_pdf = _plot_week_sweep(merged_df, output_dir=output_dir, db_scale=False)
        plot_db_png, plot_db_pdf = _plot_week_sweep(merged_df, output_dir=output_dir, db_scale=True)

    summary = {
        "output_dir": str(output_dir),
        "mltp_csv": str(mltp_csv),
        "per_cell_csv": str(per_cell_csv),
        "candidate_csv": str(candidate_csv),
        "merged_csv": None if merged_csv is None else str(merged_csv),
        "merged_csv_db": None if merged_csv_db is None else str(merged_csv_db),
        "plot_png": None if plot_png is None else str(plot_png),
        "plot_pdf": None if plot_pdf is None else str(plot_pdf),
        "plot_db_png": None if plot_db_png is None else str(plot_db_png),
        "plot_db_pdf": None if plot_db_pdf is None else str(plot_db_pdf),
        "num_selected_cells": len(selected_cells),
        "selected_cells": selected_cells,
        "raw_sources": raw_sources,
        "seed": seed,
        "k_neighbors": int(args.k_neighbors),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
