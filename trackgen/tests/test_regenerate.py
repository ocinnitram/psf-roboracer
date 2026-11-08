"""Regenerate the lab raceline and check it against the golden output of the current algorithm."""
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
GOLDEN_RACELINE = Path(__file__).resolve().parent / "golden" / MAP / "raceline.csv"


@pytest.mark.skipif(not RAW_PGM.exists(), reason="raw occupancy map is missing")
def test_raceline_parity(tmp_path: Path) -> None:
    """Regenerate raceline.csv; require last-digit agreement with the golden file."""
    from trackgen.paths import TrackPaths
    from trackgen.stages import raceline

    expected = pd.read_csv(GOLDEN_RACELINE)
    paths = TrackPaths(MAP)
    # Write to tmp so the tracked raceline.csv is left untouched.
    paths.raceline_csv = tmp_path / "raceline.csv"
    raceline.run(paths, "cw", plot=False)
    got = pd.read_csv(paths.raceline_csv)
    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected)
    for col in expected.columns:
        delta = np.abs(got[col].to_numpy(dtype=float) - expected[col].to_numpy(dtype=float))
        assert float(delta.max()) <= 0.0001 + 1e-12, f"{col} max abs {delta.max()}"
