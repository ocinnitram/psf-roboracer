"""Batched mean-action evaluation over a list of layouts."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as T

from ppo.agent import PPOAgent
from ppo.config import Config
from ppo.layouts import Layout, to_sim

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "sf", REPO / "sim" / "python"):
    text = str(extra)
    if text not in sys.path:
        sys.path.insert(0, text)

from noise import (  # noqa: E402
    LIDAR_BIAS,
    LIDAR_STOCHASTIC,
    OBS_NOISE_S_NORM,
    OBS_NOISE_VEL,
    OBS_NOISE_YAW,
    OBS_NOISE_YAW_RATE,
    ODOM_NOISE_POS_M,
    ODOM_NOISE_VEL_MPS,
)
from observation import N_OBS  # noqa: E402
from roboracer_sim import (  # noqa: E402
    COLLISION,
    COMPLETE,
    RUN_KEYS,
    VecEnv,
    car_kwargs,
)

EVAL_TIMEOUT_FACTOR = 6.0
# Plant episode cap, in seconds per second of the map's minimum-time lap. 45 s on
# the lab track (7.02 s bound), which is the cap every policy in this thesis trained
# and was evaluated under; a longer track gets the same number of laps' worth of time.
TICK_CAP_PER_OPT_S = 45.0 / 7.02


def episode_tick_cap(cfg: Config) -> int:
    """Episode cap for this map in plant ticks (1 ms)."""
    return int(round(TICK_CAP_PER_OPT_S * float(cfg.track.opt_time) * 1000.0))


def env_kwargs(cfg: Config, sim2real: bool | None = None) -> dict[str, float]:
    """Build VecEnv kwargs from car parameters plus this run's rewards and noise."""
    if sim2real is None:
        sim2real = bool(cfg.training.sim2real)
    kwargs = car_kwargs()
    if cfg.track.max_speed_mps is not None:
        kwargs["v_max"] = float(cfg.track.max_speed_mps)
    kwargs.update(
        {
            "progress_reward": float(cfg.ppo.r_progress),
            "collision_reward": float(cfg.ppo.r_collision),
            "lap_completion_reward": float(cfg.ppo.r_lap),
            "optimal_lap_time": float(cfg.track.opt_time),
            "sim2real": float(sim2real),
            "obs_noise_s_norm": OBS_NOISE_S_NORM,
            "obs_noise_yaw_rate": OBS_NOISE_YAW_RATE,
            "obs_noise_vel": OBS_NOISE_VEL,
            "obs_noise_yaw": OBS_NOISE_YAW,
            "odom_noise_pos_m": ODOM_NOISE_POS_M,
            "odom_noise_vel_mps": ODOM_NOISE_VEL_MPS,
            "lidar_bias": LIDAR_BIAS,
            "lidar_stochastic": LIDAR_STOCHASTIC,
        }
    )
    missing = [key for key in RUN_KEYS if key not in kwargs]
    if missing:
        raise ValueError(f"env kwargs missing {missing}")
    return kwargs


def blob_path(map_name: str) -> Path:
    """Return the compiled track blob path for a map."""
    return REPO / "sf" / "tracks" / map_name / f"{map_name}.blob"


def ensure_blob(map_name: str) -> Path:
    """Compile the track blob if it is missing."""
    path = blob_path(map_name)
    if path.is_file():
        return path
    import importlib.util

    loc = REPO / "sim" / "tools" / "compile_track.py"
    spec = importlib.util.spec_from_file_location("compile_track", loc)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {loc}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compile_map(map_name, plot=False)
    if not path.is_file():
        raise FileNotFoundError(f"track blob not written: {path}")
    return path


def make_venv(
    cfg: Config,
    num_envs: int,
    seed: int,
    deterministic: bool | None = None,
    sim2real: bool | None = None,
) -> VecEnv:
    """Construct a batched plant for this config."""
    if sim2real is None:
        sim2real = bool(cfg.training.sim2real)
    if deterministic is None:
        deterministic = not sim2real
    return VecEnv(
        num_envs=num_envs,
        blob_path=ensure_blob(cfg.track.map),
        kwargs=env_kwargs(cfg, sim2real),
        seed=seed,
        deterministic=deterministic,
    )


def _to_sim(layout: Layout, v_s: float = 0.0) -> list[dict[str, float]]:
    """Turn a layout into VecEnv.set_obstacles dicts."""
    return to_sim(layout, v_s=v_s)


def _mean_actions(agent: PPOAgent, states: np.ndarray) -> np.ndarray:
    """Actor-mean actions for a batch of observations."""
    st = T.tensor(states, dtype=T.float32).to(agent.actor.device)
    with T.no_grad():
        return agent.actor(st).mean.cpu().numpy().astype(np.float32, copy=False)


def _control_dt() -> float:
    """Control period in seconds from the car parameters."""
    return float(car_kwargs()["control_period"]) / 1000.0


@dataclass
class LayoutRun:
    """Mean-action outcome of one layout in a batched plant run."""

    success: bool
    collided: bool
    lap_times: list[float]
    steps: int
    terminal_v: float
    steer_rate: float
    speed_cmd_rate: float


def run_layouts(
    agent: PPOAgent,
    cfg: Config,
    layouts: Sequence[Layout],
    seed: int,
    sim2real: bool | None = None,
    num_laps: int | None = None,
) -> list[LayoutRun]:
    """Run mean-action flying laps, one env per layout."""
    n_layouts = len(layouts)
    if n_layouts == 0:
        return []
    if sim2real is None:
        sim2real = bool(cfg.training.sim2real)
    if num_laps is None:
        num_laps = int(cfg.training.num_laps)
    else:
        num_laps = int(num_laps)
    dt = _control_dt()
    max_steps = int(EVAL_TIMEOUT_FACTOR * cfg.track.opt_time * max(1, num_laps) / dt)
    venv = make_venv(
        cfg, n_layouts, seed, deterministic=not sim2real, sim2real=sim2real
    )
    try:
        venv.reset()
        tick_cap = episode_tick_cap(cfg)
        for i, layout in enumerate(layouts):
            venv.set_lap_cap(i, num_laps + 1)
            venv.set_tick_cap(i, tick_cap)
            venv.reset_env(i)
            venv.set_obstacles(i, _to_sim(layout))
        states = venv.obs.copy()

        finished = [False] * n_layouts
        success = [False] * n_layouts
        collided = [False] * n_layouts
        per_lap_times: list[list[float]] = [[] for _ in range(n_layouts)]
        prev_lap_count = [0] * n_layouts
        prev_actions = np.zeros((n_layouts, 2), dtype=np.float32)
        action_delta_sum = np.zeros((n_layouts, 2), dtype=np.float64)
        steps = np.zeros(n_layouts, dtype=np.int32)
        terminal_v = np.zeros(n_layouts, dtype=np.float64)

        for _ in range(max_steps):
            actions = np.zeros((n_layouts, 2), dtype=np.float32)
            live = [k for k in range(n_layouts) if not finished[k]]
            if live:
                actions[live] = _mean_actions(agent, states[live])
                action_delta_sum[live] += np.abs(actions[live] - prev_actions[live])
                steps[live] += 1
                prev_actions[live] = actions[live]
            venv.actions[:] = actions
            venv._lib.rr_step(venv._vec)
            states = venv.obs.copy()

            for k in range(n_layouts):
                if finished[k]:
                    continue
                if venv.terminals[k]:
                    kind = venv.done_kind(k)
                    laps = venv.done_laps(k)
                    success[k] = (
                        kind == COMPLETE
                        and laps >= num_laps + 1
                        and float(venv.rewards[k]) > 0.0
                    )
                    if success[k]:
                        per_lap_times[k].append(float(venv.done_lap_time(k)))
                    collided[k] = kind == COLLISION
                    finished[k] = True
                    continue
                terminal_v[k] = float(venv.measured_state(k)[4])
                cur_lap = int(venv._lib.rr_current_lap(venv._vec, k))
                if cur_lap > prev_lap_count[k]:
                    prev_lap_count[k] = cur_lap
                    if cur_lap >= 2:
                        last_t = float(venv._lib.rr_last_lap_time(venv._vec, k))
                        per_lap_times[k].append(last_t)
            if all(finished):
                break
    finally:
        venv.close()

    runs: list[LayoutRun] = []
    for k in range(n_layouts):
        count = int(steps[k])
        denom = max(count, 1)
        runs.append(
            LayoutRun(
                success=bool(success[k]),
                collided=bool(collided[k]),
                lap_times=per_lap_times[k],
                steps=count,
                terminal_v=float(terminal_v[k]),
                steer_rate=float(action_delta_sum[k, 0] / denom),
                speed_cmd_rate=float(action_delta_sum[k, 1] / denom),
            )
        )
    return runs


def summarise(runs: Sequence[LayoutRun], cfg: Config) -> dict[str, float]:
    """Aggregate LayoutRun records into the evaluate() metric dict."""
    n_layouts = len(runs)
    if n_layouts == 0:
        return {
            "success_rate": 0.0,
            "pace": 0.0,
            "mean_lap_time": float("nan"),
            "collisions": 0.0,
            "n_layouts": 0.0,
            "steer_rate": 0.0,
            "speed_cmd_rate": 0.0,
        }
    completed_means: list[float] = []
    for run in runs:
        if run.success and run.lap_times:
            completed_means.append(float(np.mean(run.lap_times)))
    n_success = sum(1 for run in runs if run.success)
    mean_lap_time = float(np.mean(completed_means)) if completed_means else float("nan")
    pace = float(cfg.track.opt_time) / mean_lap_time if completed_means else 0.0
    total_steps = int(sum(run.steps for run in runs))
    if total_steps > 0:
        steer_rate = float(
            sum(run.steer_rate * run.steps for run in runs) / total_steps
        )
        speed_cmd_rate = float(
            sum(run.speed_cmd_rate * run.steps for run in runs) / total_steps
        )
    else:
        steer_rate = 0.0
        speed_cmd_rate = 0.0
    return {
        "success_rate": float(n_success / n_layouts),
        "pace": pace,
        "mean_lap_time": mean_lap_time,
        "collisions": float(sum(1 for run in runs if run.collided)),
        "n_layouts": float(n_layouts),
        # Mean absolute change of the normalised commands per control tick:
        # the steering-angle command rate and the speed-command change, which
        # is the commanded acceleration.
        "steer_rate": steer_rate,
        "speed_cmd_rate": speed_cmd_rate,
    }


def evaluate(
    agent: PPOAgent,
    cfg: Config,
    layouts: Sequence[Layout],
    seed: int,
) -> dict[str, float]:
    """Run mean-action flying laps, one env per layout, and return summary metrics."""
    return summarise(run_layouts(agent, cfg, layouts, seed), cfg)


def observation_size() -> int:
    """Return the policy observation length."""
    return int(N_OBS)
