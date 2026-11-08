from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ppo_sf_roboracer = pytest.importorskip("ppo_sf_roboracer")

REPO = Path(__file__).resolve().parents[2]
RACELINE = REPO / "sf" / "tracks" / "lab-20260906" / "raceline.csv"


def test_off_centre_query_uses_exact_nearest_centreline_point() -> None:
    track = ppo_sf_roboracer.TrackHandler()
    assert track.load(str(RACELINE))
    data = track.data
    x = np.asarray(data.x)
    y = np.asarray(data.y)
    s = np.asarray(data.s)

    query_x, query_y = 4.7470, -1.4995
    exact_index = int(np.argmin((x - query_x) ** 2 + (y - query_y) ** 2))
    projected = track.to_frenet(query_x, query_y, 0.0)

    assert projected.s == s[exact_index]
