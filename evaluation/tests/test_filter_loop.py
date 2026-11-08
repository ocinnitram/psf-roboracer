"""Smoke the filter tick loop on the empty track with random actor weights."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from evaluation.common import COMPLETED, CRASHED, PARKED, REPO, TIMEOUT
from evaluation.filter import run_layout
from evaluation.log import COLUMNS, read
from ppo.config import load, resolve
from ppo.rollout import env_kwargs
from ppo.train import _make_agent
from roboracer_sim import COLLISION, COMPLETE

SO_PATH = REPO / "sim" / "build" / "libroboracer.so"
OUTCOMES = {COMPLETED, CRASHED, PARKED, TIMEOUT}


def _require_plant_and_filter(map_name: str) -> None:
    """Skip when the simulator, pybind module, or track blob is missing."""
    if not SO_PATH.is_file():
        pytest.skip(f"missing {SO_PATH}")
    matches = list((REPO / "sf" / "build").glob("ppo_sf_roboracer*.so"))
    if not matches:
        pytest.skip("missing sf/build/ppo_sf_roboracer*.so")
    blob = REPO / "sf" / "tracks" / map_name / f"{map_name}.blob"
    if not blob.is_file():
        pytest.skip(f"missing {blob}")


class _StubLib:
    """No-op plant ABI used after the first (already-terminal) step."""

    def rr_step(self, _vec) -> None:
        return None

    def rr_current_lap(self, _vec, _i: int) -> int:
        return 0

    def rr_last_lap_time(self, _vec, _i: int) -> float:
        return 0.0


class _StubVenv:
    """One-env plant that is terminal after the first step, with a fixed done_kind."""

    def __init__(self, kind: int) -> None:
        self.terminals = np.array([True])
        self.obs = np.zeros((1, 26), dtype=np.float32)
        self.actions = np.zeros((1, 2), dtype=np.float32)
        self._lib = _StubLib()
        self._vec = 0
        self._kind = kind

    def seed(self, *_args, **_kwargs) -> None:
        return None

    def reset(self) -> None:
        return None

    def set_lap_cap(self, *_args, **_kwargs) -> None:
        return None

    def set_tick_cap(self, *_args, **_kwargs) -> None:
        return None

    def reset_env(self, *_args, **_kwargs) -> None:
        return None

    def set_obstacles(self, *_args, **_kwargs) -> None:
        return None

    def raw_scan(self, _i: int, dest=None):
        arr = np.ones(811, dtype=np.float32)
        if dest is not None:
            dest[:] = arr
            return dest
        return arr

    def measured_state(self, _i: int):
        return (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    def vehicle_state(self, _i: int):
        return (0.0,) * 11

    def odom_age(self, _i: int) -> float:
        return 0.0

    def done_kind(self, _i: int) -> int:
        return self._kind

    def done_laps(self, _i: int) -> int:
        return 3 if self._kind == COMPLETE else 0

    def done_lap_time(self, _i: int) -> float:
        return 7.0

    def close(self) -> None:
        return None


class _StubTrack:
    """Frenet lookup that never depends on a loaded blob."""

    def to_frenet(self, _x, _y, _yaw):
        return SimpleNamespace(s=0.0, n=0.0)


def _patch_loop(monkeypatch: pytest.MonkeyPatch, kind: int, filt=None) -> None:
    """Drive run_layout through one stubbed terminal tick."""
    monkeypatch.setattr(
        "evaluation.filter.make_venv", lambda *_a, **_k: _StubVenv(kind)
    )
    monkeypatch.setattr(
        "evaluation.filter._stack",
        lambda *_a, **_k: (_StubTrack(), None, None, filt),
    )
    monkeypatch.setattr(
        "evaluation.filter._mean_actions",
        lambda _agent, states: np.zeros((states.shape[0], 2), dtype=np.float32),
    )
    monkeypatch.setattr("evaluation.filter._control_dt", lambda: 0.067)
    monkeypatch.setattr(
        "evaluation.filter.stack.vehicle_state", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        "evaluation.filter.stack.action", lambda delta, speed: (delta, speed)
    )
    monkeypatch.setattr(
        "evaluation.filter.stack.lidar_pose", lambda *_a, **_k: object()
    )


def _run(sf: str) -> object:
    cfg = load(resolve("empty"))
    return run_layout(
        object(),  # type:ignore[arg-type]
        cfg,
        (),
        0,
        sf=sf,
        sim2real=False,
        obstacles="detector",
        num_laps=1,
    )


def test_run_layout_empty_track_writes_log(tmp_path: Path) -> None:
    cfg = load(resolve("empty"))
    _require_plant_and_filter(cfg.track.map)
    cfg.training.run_name = "_eval_test_filter_loop"
    agent_dir = REPO / "outputs" / "trained_agents" / cfg.training.run_name
    shutil.rmtree(agent_dir, ignore_errors=True)
    log_path = tmp_path / "filter_log.csv"
    try:
        agent = _make_agent(cfg, 0)
        result = run_layout(
            agent,
            cfg,
            (),
            0,
            sf="on",
            sim2real=False,
            obstacles="detector",
            num_laps=1,
            log_path=log_path,
        )
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)

    assert result.steps > 0
    assert result.outcome in OUTCOMES
    assert log_path.is_file()
    header = log_path.read_text().splitlines()[0]
    assert tuple(header.split(",")) == COLUMNS
    data = read(log_path)
    n_rows = len(data["step"])
    assert n_rows == result.steps
    assert sum(result.outcomes.values()) == result.steps


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (COLLISION, CRASHED),
        (COMPLETE, COMPLETED),
        (0, TIMEOUT),
    ],
)
def test_plant_terminal_classifies_done_kind(
    monkeypatch: pytest.MonkeyPatch, kind: int, expected: str
) -> None:
    _patch_loop(monkeypatch, kind)
    result = _run("off")
    assert result.outcome == expected
    assert result.steps == 1


def test_shadow_set_published_uses_policy_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[object] = []

    class _Filt:
        def certify(self, _state, _tracked, _action, _include_diagnostics):
            return SimpleNamespace(
                command=SimpleNamespace(delta=0.4, v=4.0),
                outcome=SimpleNamespace(name="Qp"),
                rl_speed_capped=False,
                diagnostics=None,
            )

        def set_published(self, action) -> None:
            recorded.append(action)

    _patch_loop(monkeypatch, 0, filt=_Filt())
    result = _run("shadow")
    assert result.outcome == TIMEOUT
    assert len(recorded) == 1
    published_delta, published_v = recorded[0]
    # The stubbed policy action is zero: no steering, half of v_max.
    v_max = float(env_kwargs(load(resolve("empty")), False)["v_max"])
    assert published_delta == pytest.approx(0.0)
    assert published_v == pytest.approx(0.5 * v_max)
