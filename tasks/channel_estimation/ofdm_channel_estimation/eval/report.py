"""Minimal report writer for channel-estimation baseline evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float:
    return float(value)


def _mse_db_from_linear(value: float) -> float:
    return float(10.0 * math.log10(float(value) + 1e-12))


def _group_key(row: dict[str, Any]) -> str:
    axis = str(row.get("analysis_axis", "snr"))
    if axis == "delay_spread":
        return str(row["delay_spread"])
    if axis == "doppler":
        return str(row["doppler"])
    return str(row["snr"])


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda x: float(x)):
        group = grouped[key]
        mean_mse_linear = sum(_to_float(item["mse_linear"]) for item in group) / len(group)
        summary_rows.append(
            {
                "condition": key,
                "count": len(group),
                "mse_linear": mean_mse_linear,
                # Match the official AdaFortiTran evaluator: aggregate MSE in
                # linear scale first, then convert that mean error to dB.
                "mse_db": _mse_db_from_linear(mean_mse_linear),
                "nmse_linear": sum(_to_float(item["nmse_linear"]) for item in group) / len(group),
            }
        )

    macro_mse_linear = sum(_to_float(item["mse_linear"]) for item in rows) / len(rows)
    macro = {
        "count": len(rows),
        "mse_linear": macro_mse_linear,
        "mse_db": _mse_db_from_linear(macro_mse_linear),
        "nmse_linear": sum(_to_float(item["nmse_linear"]) for item in rows) / len(rows),
    }
    return {"macro": macro, "per_condition": summary_rows}


def write_reports(output_dir: str | Path, rows: list[dict[str, Any]], target_board: dict[str, Any] | None = None) -> dict[str, Path]:
    del target_board
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no evaluation rows to report")

    fieldnames = list(rows[0].keys())
    per_sample_csv = output_path / "per_sample.csv"
    with per_sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = _aggregate_rows(rows)

    summary_csv = output_path / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "count", "mse_linear", "mse_db", "nmse_linear"])
        writer.writeheader()
        writer.writerows(aggregate["per_condition"])
        writer.writerow(
            {
                "condition": "macro",
                "count": aggregate["macro"]["count"],
                "mse_linear": aggregate["macro"]["mse_linear"],
                "mse_db": aggregate["macro"]["mse_db"],
                "nmse_linear": aggregate["macro"]["nmse_linear"],
            }
        )

    summary_json = output_path / "summary.json"
    summary_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    summary_md = output_path / "summary.md"
    lines = [
        "| Condition | Count | MSE | MSE(dB) | NMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate["per_condition"]:
        lines.append(
            f"| {row['condition']} | {row['count']} | {row['mse_linear']:.6f} | {row['mse_db']:.3f} | {row['nmse_linear']:.6f} |"
        )
    lines.append(
        f"| macro | {aggregate['macro']['count']} | {aggregate['macro']['mse_linear']:.6f} | {aggregate['macro']['mse_db']:.3f} | {aggregate['macro']['nmse_linear']:.6f} |"
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "per_sample_csv": per_sample_csv,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "summary_md": summary_md,
    }
