"""Method-local wrapper for BeamFormer ARN training."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(_METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(_METHOD_ROOT))


if __name__ == "__main__":
    runpy.run_module("implementation.train_ARN", run_name="__main__")

