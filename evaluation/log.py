"""Filter-eval CSV: one row per control tick."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

# Unmeasured states: plant truth (true_), the filter's carried-over estimate
# (est_) and its steady-state completion (ss_) at the same tick.
HIDDEN: tuple[str, ...] = ("beta", "fy_f", "fy_r", "delta_dot", "accel")

COLUMNS: tuple[str, ...] = (
    "step", "sim_time", "lap", "s", "s_norm", "n", "x", "y", "yaw", "delta", "v", "r",
    "rl_delta", "rl_v", "sf_delta", "sf_v", "applied_delta", "applied_v",
    "outcome", "selection",
    "n_raw", "raw_x", "raw_y", "raw_yaw", "raw_length", "raw_width",
    "n_tracked", "obs_x", "obs_y", "obs_yaw", "obs_length", "obs_width", "obs_vx", "obs_vy",
    "c_rl_obstacle", "c_rl_wall", "c_rl_speed", "c_rl_lateral_stop", "c_rl_cone", "c_rl_progress",
    "c_pub_obstacle", "c_pub_wall", "c_pub_speed", "c_pub_lateral_stop", "c_pub_cone", "c_pub_progress",
    "ppo_ms", "filter_ms", "certify_ms", "n_sims", "failed",
    "speed_capped",
    *(f"{src}_{name}" for src in ("true", "est", "ss") for name in HIDDEN),
)
_COLSET = frozenset(COLUMNS)
_TEXT = frozenset({"outcome", "selection", "failed"})


class FilterLog:
    """Buffer of per-tick rows, written as CSV on demand."""

    def __init__(self, path: Path | None) -> None:
        """Buffer rows for `path`, or discard them when path is None."""
        self.path = Path(path) if path is not None else None
        self._rows: list[list[object]] = []

    def add(self, row: dict[str, object]) -> None:
        """Append one tick. Missing or extra keys raise KeyError."""
        for key in row:
            if key not in _COLSET:
                raise KeyError(key)
        values = [row[name] for name in COLUMNS]
        if self.path is not None:
            self._rows.append(values)

    def write(self) -> Path | None:
        """Write the CSV header and rows. Returns None when no path was set."""
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(COLUMNS)
            for row in self._rows:
                writer.writerow("" if isinstance(v, float) and math.isnan(v) else v for v in row)
        return self.path


def read(path) -> dict[str, np.ndarray]:
    """Load a filter log as a dict of column arrays."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = tuple(reader.fieldnames or COLUMNS)
        buckets: dict[str, list[object]] = {name: [] for name in names}
        for row in reader:
            for name in names:
                raw = row.get(name, "")
                if name in _TEXT:
                    buckets[name].append(raw)
                elif raw == "":
                    buckets[name].append(math.nan)
                else:
                    buckets[name].append(float(raw))
    return {
        name: np.asarray(values, dtype=object if name in _TEXT else np.float64)
        for name, values in buckets.items()
    }
