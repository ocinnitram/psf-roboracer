"""Run the minimum-time stage on the golden lab-20260906 raceline and check the output table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
MAP = "lab-20260906"
RAW_PGM = REPO / "sf" / "tracks" / MAP / "raw" / f"{MAP}.pgm"
# The golden file is the current raceline stage's output on this track, used as
# a regression reference.
RACELINE = Path(__file__).resolve().parent / "golden" / MAP / "raceline.csv"
# Lap time of the current solver on RACELINE at the precision opt_time is
# stored with (two decimals); the tolerance is half of that last digit.
LAP_S = 6.94
LAP_TOL_S = 0.005
COLUMNS = (
    "x",
    "y",
    "s",
    "s_norm",
    "track_yaw_rad",
    "kappa",
    "w_l",
    "w_r",
    "n_x",
    "n_y",
    "v_ref",
)


def _lap_from_csv(df: pd.DataFrame) -> float:
    """Integrate ds/v_ref around the closed mintime raceline."""
    xy = df[["x", "y"]].to_numpy(dtype=float)
    closed = np.vstack([xy, xy[:1]])
    ds = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    v = np.maximum(df["v_ref"].to_numpy(dtype=float), 1e-3)
    return float(np.sum(ds / v))


@pytest.mark.skipif(not RAW_PGM.exists(), reason="raw occupancy map is missing")
def test_mintime_lap_time(tmp_path: Path) -> None:
    """Require a v_ref table, the recorded lap time, and a lap consistent with the written profile."""
    from trackgen.paths import TrackPaths
    from trackgen.stages import mintime

    paths = TrackPaths(MAP)
    paths.raceline_csv = RACELINE
    # Write to tmp so the tracked raceline_mintime.csv is left untouched.
    paths.raceline_mintime_csv = tmp_path / "raceline_mintime.csv"
    lap = mintime.run(paths, "cw", plot=False)
    assert abs(lap - LAP_S) <= LAP_TOL_S, f"mintime lap {lap:.4f}s, expected {LAP_S}s"
    assert paths.raceline_mintime_csv.exists()
    df = pd.read_csv(paths.raceline_mintime_csv)
    assert list(df.columns) == list(COLUMNS)
    assert len(df) > 100
    csv_lap = _lap_from_csv(df)
    assert abs(csv_lap - lap) <= 0.02, f"returned {lap:.4f}s vs csv {csv_lap:.4f}s"
