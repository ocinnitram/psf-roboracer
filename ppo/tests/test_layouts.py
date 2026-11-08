from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ppo.config import ObstaclesConfig
from ppo.layouts import (
    Obstacle,
    Track,
    Zones,
    load_set,
    sample_layout,
    to_sim,
)

REPO = Path(__file__).resolve().parents[2]
MAP = "lab-20260906"


def _obs() -> ObstaclesConfig:
    return ObstaclesConfig(
        n_obstacles=2,
        min_s_distance=5.0,
        obstacle_width=0.25,
        obstacle_length=0.35,
        centre_deadzone=0.0,
        start_exclusion=5.0,
    )


def test_to_sim_keys() -> None:
    layout = (
        Obstacle(s=6.9, n=-0.05, half_length=0.175, half_width=0.125),
        Obstacle(s=20.75, n=-0.2, half_length=0.175, half_width=0.125),
    )
    packed = to_sim(layout)
    assert len(packed) == 2
    assert packed[0].keys() == {"s", "n", "v_s", "half_length", "half_width"}
    assert packed[0]["v_s"] == 0.0
    moving = to_sim(layout, v_s=0.4)
    assert moving[1]["v_s"] == 0.4
    assert moving[0]["s"] == 6.9
    assert moving[0]["half_width"] == 0.125


def test_sample_layout_bounds() -> None:
    track = Track.load(MAP)
    zones = Zones.load(MAP)
    cfg = _obs()
    half_w = cfg.obstacle_width / 2.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        layout = sample_layout(track, cfg, zones, rng)
        assert len(layout) == 2
        s_values = [item.s for item in layout]
        assert s_values[0] <= s_values[1]
        assert s_values[1] - s_values[0] >= cfg.min_s_distance
        for item in layout:
            assert item.s >= cfg.start_exclusion
            in_zone = any(lo <= item.s <= hi for lo, hi in zones.intervals)
            assert in_zone
            ndx = track.nearest_index(item.s)
            assert item.n >= float(-track.w_r[ndx] + half_w) - 1e-9
            assert item.n <= float(track.w_l[ndx] - half_w) + 1e-9
            if cfg.centre_deadzone > 0.0:
                assert abs(item.n) >= cfg.centre_deadzone
            assert item.half_length == cfg.obstacle_length / 2.0
            assert item.half_width == half_w


def test_load_set_obs_set_dev_500() -> None:
    layouts = load_set("obs-set-dev-500")
    assert len(layouts) == 500
    for layout in layouts:
        assert len(layout) == 2
    first = layouts[0]
    assert first[0].s == 6.849999999999993
    assert first[0].n == 0.3850000000000006
    assert first[0].half_length == 0.175
    assert first[0].half_width == 0.125
    assert first[1].s == 19.150000000000052
    assert first[1].n == 0.9850000000000017
    assert first[1].half_length == 0.175
    assert first[1].half_width == 0.125
    with (REPO / "evaluation" / "layouts" / "obs-set-dev-500" / "k0.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["map"] == "lab-20260906"
    obs = manifest["obstacles"][0]
    assert obs["x"] == 6.926522
    assert obs["y"] == 0.08251450000000055
    assert obs["yaw"] == -0.4084
