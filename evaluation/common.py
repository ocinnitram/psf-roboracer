"""Shared pieces of the evaluation: paths, the plant seed, and the per-layout result."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "sf", REPO / "sf" / "build", REPO / "sim" / "python"):
    text = str(extra)
    if text not in sys.path:
        sys.path.insert(0, text)

OUT_DIR = REPO / "outputs" / "eval"

# Plant seed for every evaluation. Layout k runs on env seed EVAL_SEED * 1000 + k,
# the same number the batched VecEnv gives env k. The checkpoint seed only
# selects the weights. The EVAL_SEED environment variable draws another plant
# (seed sweeps); spawned workers inherit it.
EVAL_SEED = int(os.environ.get("EVAL_SEED", "1"))

# Layout outcomes.
COMPLETED = "completed"  # every requested flying lap finished
CRASHED = "crashed"  # the plant reported a collision
PARKED = "parked"  # standstill for PARK_STANDSTILL_S after first moving
TIMEOUT = "timeout"  # plant tick cap (map-scaled, ppo.rollout.episode_tick_cap)

# Standstill detection in the tick loop.
PARK_STOP_SPEED_MPS = 0.5
PARK_STANDSTILL_S = 8.0


@dataclass
class LayoutResult:
    """What one policy run on one layout produced."""

    index: int  # layout index k; 0 on the empty track
    n_obstacles: int
    outcome: str  # COMPLETED | CRASHED | PARKED | TIMEOUT
    lap_times: list[float]  # flying laps, seconds
    steps: int  # control ticks
    sim_time: float  # seconds
    terminal_v: float  # speed at the last tick, m/s
    # Mean absolute change of the normalised commands per tick.
    steer_rate: float
    speed_cmd_rate: float
    # Filter modes only; zero / empty when the filter did not run.
    interventions: int = 0  # ticks where the applied command differs from the policy's
    # Ticks where the policy's speed was above the filter's v_max_mps and was
    # certified at it (SafetyFilterOutput.rl_speed_capped); counted in
    # interventions when the cut exceeds the speed threshold, whatever the outcome.
    speed_capped: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)  # sf.Outcome name -> ticks
    selections: dict[str, int] = field(default_factory=dict)  # sf.CommandSelection name -> ticks
    ppo_ms: dict[str, float] = field(default_factory=dict)  # timing_summary()
    filter_ms: dict[str, float] = field(default_factory=dict)
    log_path: str | None = None  # file name of the filter_log csv next to summary.json, when written

    @property
    def success(self) -> bool:
        return self.outcome == COMPLETED

    def to_dict(self) -> dict:
        return asdict(self)


def timing_summary(samples_ms: list[float]) -> dict[str, float]:
    """mean / p50 / p95 / p99 / max of per-tick wall times in milliseconds."""
    if not samples_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
