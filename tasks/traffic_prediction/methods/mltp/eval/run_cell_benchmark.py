"""Method-local wrapper for the paper-aligned cell benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[3]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from traffic_prediction_core.paper.run_cell_benchmark import main


if __name__ == "__main__":
    main()
