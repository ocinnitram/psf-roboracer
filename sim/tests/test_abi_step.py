"""ABI smoke: reset, step, dtypes, and seed reproducibility on two envs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import roboracer_sim as rr

REPO = Path(__file__).resolve().parents[2]
SO_PATH = REPO / "sim" / "build" / "libroboracer.so"
BLOB = REPO / "sf" / "tracks" / "lab-20260906" / "lab-20260906.blob"
N_OBS = 26


def _require_lib_and_blob() -> None:
    if not SO_PATH.is_file():
        pytest.skip(f"missing {SO_PATH}")
    if not BLOB.is_file():
        pytest.skip(f"missing {BLOB}")


def _env_kwargs(sim2real: float) -> dict[str, float]:
    kwargs = rr.car_kwargs()
    kwargs.update(
        {
            "sim2real": sim2real,
            "obs_noise_s_norm": 0.0006,
            "obs_noise_yaw_rate": 0.015,
            "obs_noise_vel": 0.0052,
            "obs_noise_yaw": 0.0042,
            "odom_noise_pos_m": 0.0146,
            "odom_noise_vel_mps": 0.0104,
            "lidar_bias": 0.06,
            "lidar_stochastic": 0.02,
            "progress_reward": 50.0,
            "collision_reward": -1.0,
            "lap_completion_reward": 1.0,
            "optimal_lap_time": 7.14,
        }
    )
    return kwargs


def _rollout(seed: int, steps: int, actions: np.ndarray) -> np.ndarray:
    venv = rr.VecEnv(2, BLOB, _env_kwargs(1.0), seed=seed, deterministic=False)
    try:
        obs = venv.reset()
        for _ in range(steps):
            obs, _, _ = venv.step(actions)
        return obs
    finally:
        venv.close()


def test_reset_step_shapes_and_seeds() -> None:
    _require_lib_and_blob()
    kwargs = _env_kwargs(0.0)
    venv = rr.VecEnv(2, BLOB, kwargs, seed=7, deterministic=True)
    try:
        obs = venv.reset()
        assert obs.shape == (2, N_OBS)
        assert obs.dtype == np.float32
        zeros = np.zeros((2, 2), dtype=np.float32)
        obs, reward, terminals = venv.step(zeros)
        assert obs.shape == (2, N_OBS)
        assert np.isfinite(reward).all()
        assert terminals.dtype == np.float32
        ones = np.ones((2, 2), dtype=np.float32)
        obs, reward, terminals = venv.step(ones)
        assert np.isfinite(reward).all()
        assert terminals.dtype == np.float32
    finally:
        venv.close()

    actions = np.zeros((2, 2), dtype=np.float32)
    actions[:, 0] = 0.25
    a = _rollout(123, 50, actions)
    b = _rollout(123, 50, actions)
    c = _rollout(999, 50, actions)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_tyre_params_alpha_pk_nominal() -> None:
    _require_lib_and_blob()
    venv = rr.VecEnv(1, BLOB, _env_kwargs(0.0), seed=0, deterministic=True)
    try:
        venv.reset()
        out = np.empty(3, dtype=np.float32)
        venv._lib.rr_tyre_params(venv._vec, 0, out.ctypes.data_as(rr.C_FLOAT_P))
        assert out[0] == pytest.approx(0.42, abs=1e-5)
        assert out[1] == pytest.approx(1.2, abs=1e-5)
        assert out[2] == pytest.approx(0.16, abs=1e-5)
    finally:
        venv.close()


def test_measured_state_xy_sigma_is_odom_noise_pos_m() -> None:
    _require_lib_and_blob()
    expected = 0.05  # differs from the default 0.0146, so an ignored kwarg fails
    kwargs = _env_kwargs(1.0)
    kwargs["odom_noise_pos_m"] = expected

    venv = rr.VecEnv(1, BLOB, kwargs, seed=1, deterministic=False)
    try:
        venv.reset()
        samples = np.array([venv.measured_state(0)[:2] for _ in range(4000)])
        sigma = samples.std(axis=0, ddof=1)
        assert sigma[0] == pytest.approx(expected, rel=0.08)
        assert sigma[1] == pytest.approx(expected, rel=0.08)
    finally:
        venv.close()
