"""set_published seeds the filter's previous command, which the next certify() rolls out from."""

from __future__ import annotations

from pathlib import Path

import pytest

SF_DIR = Path(__file__).resolve().parents[1]
# Top speed of the lab vehicle.
V_MAX_MPS = 4.0


def test_set_published_is_next_certify_u_prev() -> None:
    ppo_sf_roboracer = pytest.importorskip("ppo_sf_roboracer")
    track = ppo_sf_roboracer.TrackHandler()
    csv_path = SF_DIR / "tracks" / "lab-20260906" / "raceline.csv"
    assert track.load(str(csv_path))
    data = track.data
    state = ppo_sf_roboracer.VehicleState()
    state.x = data.x[0]
    state.y = data.y[0]
    state.yaw = data.track_yaw[0]
    state.v = 3.0

    published = ppo_sf_roboracer.Action()
    published.delta = 0.3
    published.v = 3.0
    u_rl = ppo_sf_roboracer.Action()
    u_rl.delta = 0.0
    u_rl.v = 3.0

    filt = ppo_sf_roboracer.SafetyFilter(track, V_MAX_MPS)
    filt.set_published(published)
    with_history = filt.certify(state, [], u_rl, True)

    fresh = ppo_sf_roboracer.SafetyFilter(track, V_MAX_MPS)
    without_history = fresh.certify(state, [], u_rl, True)

    assert with_history.diagnostics is not None
    assert without_history.diagnostics is not None
    rolled = with_history.diagnostics.predicted_states
    rolled_fresh = without_history.diagnostics.predicted_states
    assert len(rolled) >= 2 and len(rolled_fresh) >= 2
    assert abs(rolled[1].delta) > abs(rolled_fresh[1].delta) + 0.02
