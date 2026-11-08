"""Reactive baselines on synthetic corridor scans."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.reactive import FTG_LAB, DisparityExtender, FollowTheGap, make

INC = 0.0058172
N = 811
ANGLES = -0.5 * (N - 1) * INC + INC * np.arange(N)


def corridor(left: float, right: float, box: tuple[float, float, float] | None = None) -> np.ndarray:
    """Ranges to two walls parallel to the car, y = left and y = -right, and an optional box
    (x, y_min, y_max) facing the car; no-return beams are inf."""
    out = np.full(N, np.inf)
    s, c = np.sin(ANGLES), np.cos(ANGLES)
    with np.errstate(divide="ignore"):
        out = np.where(s > 1e-9, left / s, out)
        out = np.where(s < -1e-9, -right / s, out)
        if box is not None:
            x, y0, y1 = box
            d = np.where(c > 1e-9, x / c, np.inf)
            hit = (d * s >= y0) & (d * s <= y1)
            out = np.where(hit, np.minimum(out, d), out)
    return out


def ftg() -> FollowTheGap:
    return FollowTheGap(angle_inc=INC, max_steer=0.4189, **FTG_LAB)


def de() -> DisparityExtender:
    return DisparityExtender(angle_inc=INC, angle_min=ANGLES[0], max_steer=0.4189, v_top=4.0)


@pytest.mark.parametrize("driver", [ftg, de])
def test_centred_goes_straight(driver) -> None:
    steer, speed = driver()(corridor(0.6, 0.6))
    assert abs(steer) < 0.02
    assert speed > 0.0


@pytest.mark.parametrize("driver", [ftg, de])
def test_steers_back_to_the_middle(driver) -> None:
    left, _ = driver()(corridor(0.9, 0.3))
    right, _ = driver()(corridor(0.3, 0.9))
    assert left > 0.0 > right


@pytest.mark.parametrize("driver", [ftg, de])
def test_steers_round_a_box(driver) -> None:
    # A box ahead on the right of a straight: pass it on the left.
    steer, _ = driver()(corridor(0.8, 0.8, box=(1.0, -0.6, 0.0)))
    assert steer > 0.05


def test_disparity_speed_rule() -> None:
    d = de()
    assert d.speed(0.1) == 0.0
    assert d.speed(10.0) == pytest.approx(4.0)
    assert d.speed(3.0) == pytest.approx(4.0 * 0.5 / 0.6)
    assert d.speed(0.15 + 1e-9) == pytest.approx(4.0 * 0.37 / 0.6)


def test_make_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        make("pure-pursuit", 1.0, 0.4189, 4.0)
