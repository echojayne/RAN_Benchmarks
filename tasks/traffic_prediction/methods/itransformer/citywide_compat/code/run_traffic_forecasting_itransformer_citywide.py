from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from itransformer_citywide import (  # noqa: E402
    CitywideITransformerConfig,
    run_citywide_itransformer,
)


def _resolve_path(value: str | None, *, base: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base / path).resolve())


def load_config(path: Path) -> CitywideITransformerConfig:
    payload = yaml.safe_load(path.read_text())
    config_dir = path.resolve().parent
    data = payload["data"]
    training = payload["training"]
    runtime = payload["runtime"]
    benchmark_start = training.get("benchmark_start")
    return CitywideITransformerConfig(
        raw_data_dir=_resolve_path(data["raw_data_dir"], base=config_dir),
        output_dir=_resolve_path(runtime["output_dir"], base=PACKAGE_ROOT),
        cache_dir=_resolve_path(runtime["cache_dir"], base=PACKAGE_ROOT),
        exclude_cell_ids_path=_resolve_path(data.get("exclude_cell_ids_path"), base=config_dir),
        train_start=str(training.get("train_start")) if training.get("train_start") is not None else None,
        train_end=str(training.get("train_end")) if training.get("train_end") is not None else None,
        test_start=str(training.get("test_start")) if training.get("test_start") is not None else None,
        test_end=str(training.get("test_end")) if training.get("test_end") is not None else None,
        benchmark_start=str(benchmark_start) if benchmark_start is not None else None,
        train_weeks=training.get("train_weeks"),
        test_weeks=training.get("test_weeks"),
        train_ratio=training.get("train_ratio", 0.7),
        val_ratio_within_train=training.get("val_ratio_within_train", 0.1),
        context_len=training.get("context_len", 24),
        horizon=training.get("horizon", 1),
        epochs=training.get("epochs", 30),
        lr=training.get("lr", 1e-3),
        weight_decay=training.get("weight_decay", 1e-4),
        d_model=training.get("d_model", 64),
        depth=training.get("depth", 4),
        num_heads=training.get("num_heads", 4),
        ffn_dim=training.get("ffn_dim", 128),
        dropout=training.get("dropout", 0.1),
        cell_embedding_dim=training.get("cell_embedding_dim", 16),
        cell_batch_size=training.get("cell_batch_size", 2048),
        eval_cell_batch_size=training.get("eval_cell_batch_size", 4096),
        seed=training.get("seed", 42),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "config" / "itransformer_citywide.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(os.path.expandvars(args.config)).expanduser())
    run_citywide_itransformer(config=config, resume=args.resume)


if __name__ == "__main__":
    main()
