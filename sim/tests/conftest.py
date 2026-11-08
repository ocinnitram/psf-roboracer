"""Put sim/python on sys.path so tests can import roboracer_sim."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON = Path(__file__).resolve().parents[1] / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))
