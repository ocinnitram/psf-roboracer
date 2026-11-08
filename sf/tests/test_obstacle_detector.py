"""The detector reports a box seen face-on without range noise: its returns
form a line with no depth, which the square prior completes to a box."""

from __future__ import annotations

import math

import numpy as np
import pytest

ppo_sf_roboracer = pytest.importorskip("ppo_sf_roboracer")
import stack  # noqa: E402

MAP = "lab-20260906"
HALF_LENGTH_M = 0.175
HALF_WIDTH_M = 0.125


def _box(track, s: float, n: float, yaw: float):
    d = track.data
    x = float(np.interp(s, d.s, d.x) + n * np.interp(s, d.s, d.n_x))
    y = float(np.interp(s, d.s, d.y) + n * np.interp(s, d.s, d.n_y))
    rect = ppo_sf_roboracer.Rect()
    rect.cx, rect.cy, rect.yaw = x, y, yaw
    rect.length, rect.width = 2.0 * HALF_LENGTH_M, 2.0 * HALF_WIDTH_M
    return rect


def _ranges_to_box(pose, rect) -> list[float]:
    """Exact ranges of every beam to the box alone; no return elsewhere."""
    n_beams, inc = 811, 0.0058172
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    hl, hw = 0.5 * rect.length, 0.5 * rect.width
    corners = [
        (rect.cx + a * hl * c - b * hw * s, rect.cy + a * hl * s + b * hw * c)
        for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))
    ]
    ranges = []
    for i in range(n_beams):
        angle = pose.yaw + (i - 0.5 * (n_beams - 1)) * inc
        dx, dy = math.cos(angle), math.sin(angle)
        best = math.inf
        for k in range(4):
            (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
            ex, ey = x1 - x0, y1 - y0
            det = dx * (-ey) - dy * (-ex)
            if abs(det) < 1e-12:
                continue
            ox, oy = x0 - pose.x, y0 - pose.y
            t = (ox * (-ey) - oy * (-ex)) / det
            u = (dx * oy - dy * ox) / det
            if t > 0.0 and 0.0 <= u <= 1.0:
                best = min(best, t)
        ranges.append(best)
    return ranges


def test_face_on_box_without_range_noise_is_detected() -> None:
    track = stack.load_track(MAP)
    detector = stack.make_detector(MAP, stack.detector_config(sim2real=False), track)
    d = track.data
    s_car, s_box = 2.0, 5.1
    yaw_box = float(np.interp(s_box, d.s, d.track_yaw))
    box = _box(track, s_box, 0.0, yaw_box)
    # The LiDAR on the line through the box centre, looking straight at its face.
    back = s_box - s_car
    pose = ppo_sf_roboracer.Pose2D()
    pose.x = box.cx - back * math.cos(yaw_box)
    pose.y = box.cy - back * math.sin(yaw_box)
    pose.yaw = yaw_box
    ranges = _ranges_to_box(pose, box)
    assert sum(math.isfinite(r) for r in ranges) >= 10
    raw = detector.detect(ranges, pose)
    near = [it for it in raw.items if math.hypot(it.rect.cx - box.cx, it.rect.cy - box.cy) < 0.3]
    assert raw.valid and near, "face-on box without range noise was dropped"
