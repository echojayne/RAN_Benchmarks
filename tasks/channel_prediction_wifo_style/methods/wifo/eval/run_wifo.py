"""Method-local wrapper for the original WiFo entrypoint."""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

_METHOD_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _METHOD_ROOT / "wifo_prediction"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


def _print_help_if_requested() -> bool:
    if not {"-h", "--help"}.intersection(sys.argv[1:]):
        return False
    parser = argparse.ArgumentParser(
        description="Run the original WiFo CLI entrypoint. Full execution requires dependencies from requirements.txt."
    )
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--size", default="middle")
    parser.add_argument("--mask_strategy_random", default="batch")
    parser.add_argument("--mask_strategy", default="random")
    parser.add_argument("--dataset", default="DS1")
    parser.add_argument("--file_load_path", default="")
    parser.add_argument("--few_ratio", type=float, default=0.5)
    parser.add_argument("--t_patch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pos_emb", default="SinCos")
    parser.print_help()
    return True


if __name__ == "__main__":
    if _print_help_if_requested():
        raise SystemExit(0)
    from main import main

    os.chdir(_SOURCE_ROOT)
    main()
