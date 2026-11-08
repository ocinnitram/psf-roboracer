"""The spliner on a synthetic circular line."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.spliner import (
    GB_TRACK,
    OVERTAKE,
    TRAILING,
    FrenetObstacle,
    Line,
    Spliner,
    frenet_obstacle,
)

R = 5.0  # radius of a counter-clockwise circle, kappa = +1/R


@pytest.fixture
def line(tmp_path) -> Line:
    s = np.arange(0.0, 2 * math.pi * R, 0.05)
    a = s / R
    rows = np.c_[R * np.cos(a), R * np.sin(a), s, s / s[-1], a + math.pi / 2, np.full_like(s, 1 / R),
                 np.full_like(s, 0.9), np.full_like(s, 0.9), -np.cos(a), -np.sin(a)]
    path = tmp_path / "raceline.csv"
    np.savetxt(path, rows, delimiter=",", header="x,y,s,s_norm,track_yaw_rad,kappa,w_l,w_r,n_x,n_y", comments="")
    return Line(path)


def spliner(line: Line, **kw) -> Spliner:
    return Spliner(line=line, max_steer=0.4189, wheelbase=0.3302, v_max=4.0, **kw)


def test_frenet_round_trip(line):
    assert line.length == pytest.approx(2 * math.pi * R, abs=1e-3)
    s = np.linspace(0.0, line.length, 37, endpoint=False)
    x, y = line.to_cartesian(s, 0.3)
    # Left of a counter-clockwise circle is inwards.
    assert np.hypot(x, y) == pytest.approx(R - 0.3, abs=1e-3)
    s2, d2 = line.to_frenet(x, y)
    assert (s2 - s + 1) % line.length - 1 == pytest.approx(0.0, abs=1e-3)
    assert d2 == pytest.approx(0.3, abs=1e-3)


def test_frenet_obstacle_extent(line):
    x, y = line.to_cartesian(3.0, -0.2)
    yaw = 3.0 / R + math.pi / 2
    ob = frenet_obstacle(line, float(x), float(y), yaw, 0.35, 0.25)
    assert ob.s_center == pytest.approx(3.0, abs=1e-3)
    assert ob.d_center == pytest.approx(-0.2, abs=1e-3)
    assert ob.d_left == pytest.approx(-0.075, abs=0.01)
    assert ob.d_right == pytest.approx(-0.325, abs=0.01)
    assert ob.s_end - ob.s_start == pytest.approx(0.35, abs=0.02)


def test_speed_profile_bounded_by_lateral_acceleration(line):
    v = line.speed_profile(10.0, 4.0, 3.0)
    assert v == pytest.approx(math.sqrt(4.0 * R), rel=1e-6)


def obstacle(s: float, d: float, half_w: float = 0.125, half_l: float = 0.175) -> FrenetObstacle:
    return FrenetObstacle(s - half_l, s + half_l, s, d + half_w, d - half_w, d)


def test_plan_passes_on_the_side_with_more_room(line):
    sp = spliner(line, evasion_dist=0.4)
    ev = sp.plan([obstacle(10.0, 0.1)], s=2.0, d=0.0, vs=0.0)
    assert ev is not None and ev.side == "right"
    apex = ev.d[np.argmin(np.abs(ev.s - 10.0))]
    assert apex == pytest.approx(0.1 - 0.125 - 0.4, abs=0.02)
    assert np.all(ev.d <= 1e-9)


def test_plan_ignores_far_and_off_line_obstacles(line):
    sp = spliner(line)
    assert sp.plan([obstacle(20.0, 0.0)], s=2.0, d=0.0, vs=0.0) is None  # beyond the lookahead
    assert sp.plan([obstacle(5.0, 0.5)], s=2.0, d=0.0, vs=0.0) is None  # off the line
    assert sp.rejected == ""


def test_plan_rejects_a_spline_into_the_wall(line):
    sp = spliner(line, evasion_dist=0.65)  # apex 0.125 + 0.65 beyond 0.9 - 0.2
    assert sp.plan([obstacle(10.0, 0.0)], s=2.0, d=0.0, vs=0.0) is None
    assert sp.rejected == "bounds"


def test_states_and_commands(line):
    sp = spliner(line, evasion_dist=0.4, splini_hyst_timer=0.0)
    x, y = line.to_cartesian(2.0, 0.0)
    yaw = 2.0 / R + math.pi / 2
    steer, v = sp(float(x), float(y), yaw, 2.0, [])
    assert sp.state == GB_TRACK
    assert steer > 0.0  # a left-hand circle
    assert v == pytest.approx(4.0)  # v_max, below sqrt(a_lat R)
    obs = [obstacle(6.0, 0.0)]
    sp(float(x), float(y), yaw, 2.0, obs)
    assert sp.state == TRAILING
    sp(float(x), float(y), yaw, 2.0, obs)
    assert sp.state == OVERTAKE


def test_map_lookup_inverts_the_table():
    from evaluation.common import REPO
    from evaluation.spliner import SteerLookup

    lut = SteerLookup(REPO / "evaluation" / "data" / "map_lut.csv")
    j = int(np.abs(lut.v - 2.0).argmin())
    i = 20
    a_y = float(lut.a_y[i, j])
    assert lut(a_y, 2.0) == pytest.approx(lut.steer[i], abs=1e-9)
    assert lut(-a_y, 2.0) == pytest.approx(-lut.steer[i], abs=1e-9)
    # Halfway between two rows interpolates between their steering commands.
    mid = 0.5 * (lut.a_y[i, j] + lut.a_y[i + 1, j])
    assert lut.steer[i] < lut(mid, 2.0) < lut.steer[i + 1]
    # More lateral acceleration needs more steering, and more speed needs less for the same a_y.
    assert lut(1.0, 2.0) < lut(2.0, 2.0)
    assert lut(2.0, 3.0) < lut(2.0, 2.0)
