"""Seed-mean pace ignores seeds with no completions; JSON writes null for NaN."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evaluation.common import COMPLETED, CRASHED, LayoutResult
from evaluation.report import _mean_block, summarise, write
from ppo.config import load, resolve


def _result(*, outcome: str, lap_times: list[float]) -> LayoutResult:
    return LayoutResult(
        index=0,
        n_obstacles=0,
        outcome=outcome,
        lap_times=lap_times,
        steps=10,
        sim_time=1.0,
        terminal_v=0.0,
        steer_rate=0.0,
        speed_cmd_rate=0.0,
    )


def test_seed_mean_pace_ignores_uncompleted_seed(tmp_path: Path) -> None:
    cfg = load(resolve("empty"))
    lap = float(cfg.track.opt_time) / 0.93
    done = _result(outcome=COMPLETED, lap_times=[lap, lap])
    fail = _result(outcome=CRASHED, lap_times=[])
    completed = summarise([done], cfg)
    failed = summarise([fail], cfg)
    assert completed["pace"] == pytest.approx(0.93)
    assert math.isnan(failed["pace"])
    assert math.isnan(failed["mean_lap_time"])
    mean = _mean_block([completed, failed])
    assert mean["pace"] == pytest.approx(0.93)

    path = write(tmp_path, cfg, {}, {1: [done], 2: [fail]})
    text = path.read_text()
    assert "NaN" not in text
    data = json.loads(text)
    assert data["seeds"]["2"]["pace"] is None
    assert data["seeds"]["2"]["mean_lap_time"] is None
    assert data["mean"]["pace"] == pytest.approx(0.93)
