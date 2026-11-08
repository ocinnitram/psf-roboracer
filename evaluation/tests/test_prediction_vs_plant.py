"""Check the filter's stage-1 predicted state against the plant after one control period.

After a warmup of identical held commands, `SafetyFilter.certify(...,
include_diagnostics=True)` exports a rollout whose second state is one control
period of the vehicle model under that command. One more plant step with the
same command must land within 5 cm in position and 0.2 m/s in speed. Later
rollout stages use the backup command and are not compared.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from observation import action_from_command
from roboracer_sim import car_kwargs

from evaluation.common import REPO
from ppo.config import load, resolve
from ppo.rollout import make_venv

SO_PATH = REPO / "sim" / "build" / "libroboracer.so"
# Enough held-command ticks to settle above the kinematic speed floor while
# the car is still on the empty track.
WARMUP_TICKS = 18
STEER_CMD_RAD = 0.0
SPEED_CMD_MPS = 2.0
POS_TOL_M = 0.05
SPEED_TOL_MPS = 0.2


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


def _wrap_yaw(yaw: float) -> float:
    """Wrap a heading to (-pi, pi]."""
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def test_stage1_prediction_matches_one_plant_period() -> None:
    cfg = load(resolve("empty"))
    _require_plant_and_filter(cfg.track.map)
    import ppo_sf_roboracer as sf

    kwargs = car_kwargs()
    action = np.asarray(
        [
            action_from_command(
                STEER_CMD_RAD, SPEED_CMD_MPS, kwargs["s_max"], kwargs["v_max"]
            )
        ],
        dtype=np.float32,
    )
    venv = make_venv(cfg, 1, seed=1, deterministic=True, sim2real=False)
    try:
        venv.reset()
        for _ in range(WARMUP_TICKS):
            _obs, _reward, terminals = venv.step(action)
            if bool(terminals[0]):
                pytest.fail(f"plant terminated during the {WARMUP_TICKS}-tick warmup")
        measured = venv.measured_state(0)
        track = sf.TrackHandler()
        track.load(str(REPO / "sf" / "tracks" / cfg.track.map / "raceline.csv"))
        filt = sf.SafetyFilter(track, float(kwargs["v_max"]))
        state = sf.VehicleState()
        state.x = float(measured[0])
        state.y = float(measured[1])
        state.yaw = _wrap_yaw(float(measured[2]))
        state.delta = float(measured[3])
        state.v = float(measured[4])
        state.r = float(measured[5])
        command = sf.Action()
        command.delta = STEER_CMD_RAD
        command.v = SPEED_CMD_MPS
        out = filt.certify(state, [], command, include_diagnostics=True)
        assert out.diagnostics is not None
        predicted = out.diagnostics.predicted_states
        assert len(predicted) >= 2

        pos0 = math.hypot(
            predicted[0].x - float(measured[0]), predicted[0].y - float(measured[1])
        )
        speed0 = abs(predicted[0].v - float(measured[4]))
        assert pos0 < 1e-6
        assert speed0 < 1e-6

        venv.step(action)
        plant = venv.measured_state(0)
        pos_err = math.hypot(
            predicted[1].x - float(plant[0]), predicted[1].y - float(plant[1])
        )
        speed_err = abs(predicted[1].v - float(plant[4]))
        assert pos_err < POS_TOL_M
        assert speed_err < SPEED_TOL_MPS
    finally:
        venv.close()
