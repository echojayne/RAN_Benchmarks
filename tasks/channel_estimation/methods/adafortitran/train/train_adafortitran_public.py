"""Run the official public AdaFortiTran training script on the benchmark data line."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def default_upstream_root() -> Path:
    return Path(__file__).resolve().parents[1] / "public_compat" / "adafortitran_public"


def default_data_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "adafortitran_benchmark.yaml"


def default_output_root() -> Path:
    return Path(os.environ.get("RAN_BENCHMARK_OUTPUT_ROOT", "outputs")) / "official_public_adafortitran"


def default_log_root() -> Path:
    return Path(os.environ.get("RAN_BENCHMARK_LOG_ROOT", "logs")) / "official_public_adafortitran"


def resolve_split_dirs(data_config_path: Path) -> tuple[Path, Path]:
    cfg = load_yaml(data_config_path)
    dataset_cfg = cfg["dataset"]
    root_dir = Path(os.path.expandvars(str(dataset_cfg["root_dir"]))).expanduser()
    return root_dir / str(dataset_cfg["train_dir"]), root_dir / str(dataset_cfg["val_dir"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(default_data_config()))
    parser.add_argument("--train-set", default="")
    parser.add_argument("--val-set", default="")
    parser.add_argument("--upstream-root", default=str(default_upstream_root()))
    parser.add_argument("--output-root", default=str(default_output_root()))
    parser.add_argument("--log-root", default=str(default_log_root()))
    parser.add_argument("--exp-id", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--save-every-n-epochs", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--print-upstream-argv", action="store_true")
    return parser.parse_args()


def build_upstream_argv(args: argparse.Namespace) -> list[str]:
    upstream_root = Path(os.path.expandvars(str(args.upstream_root))).expanduser().resolve()
    data_config_path = Path(os.path.expandvars(str(args.data_config))).expanduser().resolve()
    if args.train_set and args.val_set:
        train_set = Path(os.path.expandvars(str(args.train_set))).expanduser().resolve()
        val_set = Path(os.path.expandvars(str(args.val_set))).expanduser().resolve()
    elif args.train_set or args.val_set:
        raise ValueError("--train-set and --val-set must be provided together")
    else:
        train_set, val_set = resolve_split_dirs(data_config_path)
    exp_id = args.exp_id or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(os.path.expandvars(str(args.output_root))).expanduser().resolve()
    log_root = Path(os.path.expandvars(str(args.log_root))).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    argv = [
        "train.py",
        "--model_name",
        "adafortitran",
        "--system_config_path",
        str(upstream_root / "config/system_config.yaml"),
        "--model_config_path",
        str(upstream_root / "config/adafortitran.yaml"),
        "--train_set",
        str(train_set),
        "--val_set",
        str(val_set),
        "--exp_id",
        exp_id,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.learning_rate),
        "--max_epoch",
        str(args.max_epoch),
        "--patience",
        str(args.patience),
        "--weight_decay",
        str(args.weight_decay),
        "--num_workers",
        str(args.num_workers),
        "--tensorboard_log_dir",
        str(output_root),
        "--python_log_dir",
        str(log_root),
        "--save_best_only",
    ]
    if args.gradient_clip_val > 0.0:
        argv.extend(["--gradient_clip_val", str(args.gradient_clip_val)])
    if args.save_every_n_epochs > 0:
        argv.extend(["--save_every_n_epochs", str(args.save_every_n_epochs)])
    if args.resume_from_checkpoint:
        resume_path = Path(os.path.expandvars(str(args.resume_from_checkpoint))).expanduser().resolve()
        argv.extend(["--resume_from_checkpoint", str(resume_path)])
    if args.use_mixed_precision:
        argv.append("--use_mixed_precision")
    return argv


def main() -> int:
    args = parse_args()
    upstream_root = Path(os.path.expandvars(str(args.upstream_root))).expanduser().resolve()
    if not upstream_root.exists():
        raise FileNotFoundError(f"upstream root not found: {upstream_root}")

    sys.path.insert(0, str(upstream_root))
    from src.train import main as upstream_main

    upstream_argv = build_upstream_argv(args)
    if args.print_upstream_argv:
        print(" ".join(upstream_argv))

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        sys.argv = upstream_argv
        os.chdir(upstream_root)
        upstream_main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
