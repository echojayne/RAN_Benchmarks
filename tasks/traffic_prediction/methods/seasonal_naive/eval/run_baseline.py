"""Method-local wrapper for seasonal naive baseline evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[3]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from traffic_prediction_core.eval.run_baseline import main


if __name__ == "__main__":
    raise SystemExit(main())

