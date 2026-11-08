"""rr_set_track swaps the mmap'd blob per env and error paths return -1."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import roboracer_sim as rr

REPO = Path(__file__).resolve().parents[2]
SO_PATH = REPO / "sim" / "build" / "libroboracer.so"
MAP = "lab-20260906"
TRACK = REPO / "sf" / "tracks" / MAP
BLOB = TRACK / f"{MAP}.blob"
COMPILE = REPO / "sim" / "tools" / "compile_track.py"


def _require_lib_and_blob() -> None:
    if not SO_PATH.is_file():
        pytest.skip(f"missing {SO_PATH}")
    if not BLOB.is_file():
        pytest.skip(f"missing {BLOB}")
    if not COMPILE.is_file():
        pytest.skip(f"missing {COMPILE}")


def _compile_track():
    spec = importlib.util.spec_from_file_location("compile_track", COMPILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_kwargs() -> dict[str, float]:
    kwargs = rr.car_kwargs()
    kwargs.update(
        {
            "sim2real": 0.0,
            "obs_noise_s_norm": 0.0,
            "obs_noise_yaw_rate": 0.0,
            "obs_noise_vel": 0.0,
            "obs_noise_yaw": 0.0,
            "odom_noise_pos_m": 0.0,
            "odom_noise_vel_mps": 0.0,
            "lidar_bias": 0.0,
            "lidar_stochastic": 0.0,
            "progress_reward": 50.0,
            "collision_reward": -1.0,
            "lap_completion_reward": 1.0,
            "optimal_lap_time": 7.14,
        }
    )
    return kwargs


def _make_obstacle_blob(out: Path) -> Path:
    ct = _compile_track()
    yaml_path = TRACK / "raw" / f"{MAP}.yaml"
    cfg = ct.load_map_yaml(yaml_path)
    resolution = float(cfg["resolution"])
    origin = cfg["origin"][:2]
    image_path = TRACK / "raw" / cfg["image"]
    free = ct.load_occupancy(image_path, cfg)
    rline = ct.load_raceline(TRACK / "raceline.csv")
    yaw = float(rline[0, 4])
    cx = float(rline[0, 0]) + 0.8 * math.cos(yaw)
    cy = float(rline[0, 1]) + 0.8 * math.sin(yaw)
    height, width = free.shape
    yy, xx = np.mgrid[0:height, 0:width]
    wx = float(origin[0]) + xx * resolution
    wy = float(origin[1]) + yy * resolution
    free[(wx - cx) ** 2 + (wy - cy) ** 2 < 0.25**2] = 0
    edt = ct.build_signed_edt(free, resolution)
    ct.pack_blob(out, edt, rline, resolution, origin)
    return out


def _rollout(venv: rr.VecEnv, steps: int) -> tuple[np.ndarray, int | None]:
    obs = venv.reset()
    trace = [obs[0].copy()]
    done_step: int | None = None
    t = 0.0
    for i in range(steps):
        action = np.array([[0.25 * np.sin(2.0 * np.pi * t / 3.0), 0.0]], dtype=np.float32)
        obs, _, terminals = venv.step(action)
        t += 0.067
        if terminals[0]:
            done_step = i
            break
        trace.append(obs[0].copy())
    return np.array(trace), done_step


def test_blob_switch_changes_then_restores() -> None:
    _require_lib_and_blob()
    with TemporaryDirectory() as tmp:
        obstacle_blob = _make_obstacle_blob(Path(tmp) / "obstacle.blob")
        venv = rr.VecEnv(1, BLOB, _env_kwargs(), seed=0, deterministic=True)
        try:
            base1, base1_done = _rollout(venv, 150)

            assert venv.set_track(0, obstacle_blob) == 0
            obst, _obst_done = _rollout(venv, 150)

            assert venv.set_track(0, BLOB) == 0
            base2, base2_done = _rollout(venv, 150)

            bad_idx = venv.set_track(5, BLOB)
            bad_path = venv.set_track(0, "/nonexistent/blob.bin")
            assert venv.set_track(0, BLOB) == 0
        finally:
            venv.close()

    assert base1.shape != obst.shape or not np.array_equal(base1, obst)
    assert base1_done == base2_done
    assert base1.shape == base2.shape
    assert np.array_equal(base1, base2)
    assert bad_idx == -1 and bad_path == -1
