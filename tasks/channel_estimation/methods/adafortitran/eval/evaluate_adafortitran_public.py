"""Run the official public AdaFortiTran evaluation script on the benchmark test tree."""

from __future__ import annotations

import argparse
import os
import sys
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


def resolve_test_root(data_config_path: Path) -> Path:
    cfg = load_yaml(data_config_path)
    dataset_cfg = cfg["dataset"]
    root_dir = Path(os.path.expandvars(str(dataset_cfg["root_dir"]))).expanduser()
    test_dir = Path(str(dataset_cfg["test_dir"]))
    return root_dir / test_dir.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-config", default=str(default_data_config()))
    parser.add_argument("--upstream-root", default=str(default_upstream_root()))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-upstream-argv", action="store_true")
    return parser.parse_args()


def build_upstream_argv(args: argparse.Namespace) -> list[str]:
    upstream_root = Path(os.path.expandvars(str(args.upstream_root))).expanduser().resolve()
    checkpoint_path = Path(os.path.expandvars(str(args.checkpoint_path))).expanduser().resolve()
    test_root = resolve_test_root(Path(os.path.expandvars(str(args.data_config))).expanduser().resolve())
    output_dir = (
        Path(os.path.expandvars(str(args.output_dir))).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / "official_public_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        "evaluate.py",
        "--checkpoint_path",
        str(checkpoint_path),
        "--test_set",
        str(test_root),
        "--batch_size",
        str(args.batch_size),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
    ]


def main() -> int:
    args = parse_args()
    upstream_root = Path(os.path.expandvars(str(args.upstream_root))).expanduser().resolve()
    if not upstream_root.exists():
        raise FileNotFoundError(f"upstream root not found: {upstream_root}")

    sys.path.insert(0, str(upstream_root))
    from src.evaluate import main as upstream_main

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
