"""One-layout tick loop with optional detector, tracker, and filter."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import observation
import ppo_sf_roboracer as ppo_sf
import stack
from roboracer_sim import COLLISION, COMPLETE, Renderer

from evaluation.common import (
    COMPLETED,
    CRASHED,
    EVAL_SEED,
    PARK_STANDSTILL_S,
    PARK_STOP_SPEED_MPS,
    PARKED,
    REPO,
    TIMEOUT,
    LayoutResult,
    timing_summary,
)
from evaluation.log import HIDDEN, FilterLog
from evaluation.reactive import make as make_reactive
from evaluation.spliner import frenet_obstacle
from evaluation.spliner import make as make_spliner
from ppo.agent import PPOAgent
from ppo.config import Config
from ppo.layouts import Layout, to_sim
from ppo.rollout import (
    _control_dt,
    _mean_actions,
    car_kwargs,
    env_kwargs,
    episode_tick_cap,
    make_venv,
)

INTERVENTION_EPS_STEER_RAD = 0.01
INTERVENTION_EPS_SPEED_MPS = 0.05
_BARRIERS = ("obstacle", "wall", "speed", "lateral_stop", "cone", "progress")


def _wrap_yaw(yaw: float) -> float:
    """Wrap heading into (-pi, pi]."""
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _stack(
    cfg: Config,
    sim2real: bool,
    sf_mode: str,
    obstacles: str,
    filter_v_max: float,
    perceive: bool = False,
):
    """Load the track and, when the filter runs, detector, tracker, and filter.
    perceive: build the detector and tracker without the filter."""
    track = stack.load_track(cfg.track.map)
    detector = tracker = filt = None
    if sf_mode != "off":
        filt = ppo_sf.SafetyFilter(track, filter_v_max)
    if sf_mode != "off" or perceive:
        if obstacles == "detector":
            config = stack.detector_config(sim2real)
            detector = stack.make_detector(cfg.track.map, config, track)
            tracker = ppo_sf.ObstacleTracker()
    return track, detector, tracker, filt


def _truth_rects(venv) -> list:
    """Build filter rectangles from the plant's live obstacles."""
    rects = []
    for row in venv.get_obstacles(0):
        rect = ppo_sf.Rect()
        rect.cx, rect.cy, rect.yaw = row["cx"], row["cy"], row["yaw"]
        rect.length, rect.width = 2.0 * row["half_length"], 2.0 * row["half_width"]
        rect.vx, rect.vy = row["vx"], row["vy"]
        rects.append(rect)
    return rects


def _nearest(rects, x: float, y: float):
    """Return the rectangle closest to the car, or None."""
    return (
        min(rects, key=lambda r: (r.cx - x) ** 2 + (r.cy - y) ** 2) if rects else None
    )


def _perceive(obstacles: str, detector, tracker, venv, scan, pose):
    """Raw detections and tracked (or truth) rectangles for this tick."""
    if obstacles == "truth":
        return [], _truth_rects(venv)
    if detector is None or tracker is None:
        return [], []
    return stack.perceive(detector, tracker, scan, pose)


def _rect_log(prefix: str, rect, with_v: bool) -> dict[str, object]:
    """CSV fields for one box; NaNs when missing."""
    keys = [
        f"{prefix}x",
        f"{prefix}y",
        f"{prefix}yaw",
        f"{prefix}length",
        f"{prefix}width",
    ]
    if with_v:
        keys += [f"{prefix}vx", f"{prefix}vy"]
    if rect is None:
        return {key: math.nan for key in keys}
    vals = [rect.cx, rect.cy, rect.yaw, rect.length, rect.width]
    if with_v:
        vals += [rect.vx, rect.vy]
    return {key: float(val) for key, val in zip(keys, vals)}


def _constraint_log(diag) -> dict[str, object]:
    """RL and published barrier rows for the CSV."""
    nan6 = [math.nan] * 6
    rl = (
        nan6
        if diag is None
        else [float(diag.constraint_values_rl[i]) for i in range(6)]
    )
    pub = (
        nan6
        if diag is None
        else [float(diag.constraint_values_published[i]) for i in range(6)]
    )
    row: dict[str, object] = {}
    for name, a, b in zip(_BARRIERS, rl, pub):
        row[f"c_rl_{name}"] = a
        row[f"c_pub_{name}"] = b
    return row


def _hidden_log(truth, diag) -> dict[str, object]:
    """Plant, carried-over and steady-state hidden states; NaNs without a filter."""
    # rr_vehicle_state order: ..., beta, accel, delta_dot, fy_f, fy_r.
    true = dict(zip(("beta", "accel", "delta_dot", "fy_f", "fy_r"), map(float, truth[6:])))
    row: dict[str, object] = {f"true_{name}": true[name] for name in HIDDEN}
    for prefix, hidden in (
        ("est", None if diag is None else diag.hidden_estimate),
        ("ss", None if diag is None else diag.hidden_steady_state),
    ):
        for name in HIDDEN:
            row[f"{prefix}_{name}"] = math.nan if hidden is None else float(getattr(hidden, name))
    return row


def _chip(name: str) -> int:
    """Renderer chip: 0 passthrough, 1 correction, 2 uncertified."""
    if name == "Uncertified":
        return 2
    return 1 if name in ("Qp", "Segment", "Backup") else 0


def _render(
    renderer, venv, tracked, outcome, predicted, x, y, dt, slowdown, t0
) -> None:
    """Draw overlays and pace wall time when a window is open."""
    if renderer is None:
        return
    renderer.set_det_boxes([[r.cx, r.cy, r.yaw, r.length, r.width] for r in tracked])
    chip = _chip(outcome)
    renderer.set_safety(chip)
    renderer.set_filter_path([[st.x, st.y] for st in predicted])
    renderer.push_trace(x, y, chip)
    renderer.render(venv, 0)
    leftover = slowdown * dt - (time.perf_counter() - t0)
    if leftover > 0.0:
        time.sleep(leftover)


def run_layout(
    agent: PPOAgent | None,
    cfg: Config,
    layout: Layout,
    index: int,
    *,
    sf: str,
    sim2real: bool,
    obstacles: str,
    num_laps: int,
    log_path: Path | None = None,
    render: bool = False,
    slowdown: float = 1.0,
    driver: str = "policy",
    speed_scale: float = 1.0,
    driver_params: dict | None = None,
) -> LayoutResult:
    """Run one layout through the driver (the policy, a reactive baseline on the
    scan, or the spliner on the tracked obstacles), optional filter, and plant.
    The driver's command is logged as rl_*."""
    if sf not in ("on", "shadow", "off"):
        raise ValueError(f"sf must be on, shadow, or off, got {sf!r}")
    if obstacles not in ("detector", "truth"):
        raise ValueError(f"obstacles must be detector or truth, got {obstacles!r}")

    car = env_kwargs(cfg, sim2real)
    steer_lock, v_max = float(car["s_max"]), float(car["v_max"])
    lidar_dist, dt, length = (
        float(car["lidar_mount_offset"]),
        _control_dt(),
        float(cfg.track.length),
    )
    use_filter = sf != "off"
    reactive = spliner = None
    if driver in ("spliner", "spliner-pp"):
        spliner = make_spliner(
            REPO / "sf" / "tracks" / cfg.track.map / "raceline.csv",
            speed_scale,
            steer_lock,
            v_max,
            dt,
            controller="pp" if driver == "spliner-pp" else "map",
            **(driver_params or {}),
        )
    elif driver != "policy":
        reactive = make_reactive(
            driver, speed_scale, steer_lock, v_max, **(driver_params or {})
        )
    if driver == "policy" and agent is None:
        raise ValueError("the policy driver needs an agent")
    # Top speed the certificate is built for: the horizon must reach a stop
    # from it and commands are capped at it. Per track; the vehicle's own cap
    # when the track does not set one.
    filter_v_max = (
        float(cfg.track.filter_v_max_mps)
        if cfg.track.filter_v_max_mps is not None
        else float(car_kwargs()["v_max"])
    )
    perceive = use_filter or spliner is not None
    track, detector, tracker, filt = _stack(
        cfg, sim2real, sf, obstacles, filter_v_max, perceive=spliner is not None
    )
    log = FilterLog(log_path)
    venv = make_venv(cfg, 1, EVAL_SEED, sim2real=sim2real)
    renderer = None
    try:
        venv.seed(0, EVAL_SEED * 1000 + index)
        venv.set_lap_cap(0, num_laps + 1)
        venv.set_tick_cap(0, episode_tick_cap(cfg))
        # Same reset sequence as ppo.rollout.run_layouts (reset, then reset_env)
        # so a sim2real noise stream matches the batched --sf off run.
        venv.reset()
        venv.reset_env(0)
        venv.set_obstacles(0, to_sim(layout))
        if render:
            renderer = Renderer()
            renderer.set_trace(True)
        scan = venv.raw_scan(0)
        lap_times: list[float] = []
        outcomes: Counter[str] = Counter()
        selections: Counter[str] = Counter()
        ppo_times: list[float] = []
        filter_times: list[float] = []
        prev_action = np.zeros(2, dtype=np.float32)
        delta_sum = np.zeros(2, dtype=np.float64)
        sim_time = 0.0
        steps = 0
        terminal_v = 0.0
        interventions = 0
        speed_capped = 0
        prev_lap = 0
        first_boundary = False
        has_moved = False
        standstill_s = 0.0
        parked = crashed = completed = False

        while True:
            t0 = time.perf_counter()
            x, y, yaw_raw, delta, v, r = map(float, venv.measured_state(0))
            yaw = _wrap_yaw(yaw_raw)
            terminal_v = v
            frenet = track.to_frenet(x, y, yaw)
            # The scan the detector and a reactive driver see this period.
            if perceive or reactive is not None:
                (venv.measured_scan if sim2real else venv.raw_scan)(0, scan)
            raw_items: list = []
            tracked: list = []
            perceive_ms = 0.0
            if perceive:
                t_p = time.perf_counter()
                raw_items, tracked = _perceive(
                    obstacles,
                    detector,
                    tracker,
                    venv,
                    scan,
                    stack.lidar_pose(x, y, yaw, v, r, venv.odom_age(0), lidar_dist),
                )
                perceive_ms = (time.perf_counter() - t_p) * 1000.0
            t_ppo = time.perf_counter()
            if spliner is not None:
                frenet_obs = [
                    frenet_obstacle(spliner.line, o.cx, o.cy, o.yaw, o.length, o.width)
                    for o in tracked
                ]
                rl_delta, rl_v = spliner(x, y, yaw, v, frenet_obs)
                selection_name = spliner.state
                failed = spliner.rejected
            elif reactive is not None:
                rl_delta, rl_v = reactive(scan)
            else:
                act = _mean_actions(agent, venv.obs[0][None])[0]
                rl_delta, rl_v = observation.command_from_action(act, steer_lock, v_max)
            if driver != "policy":
                rl_delta = float(np.clip(rl_delta, -steer_lock, steer_lock))
                rl_v = float(np.clip(rl_v, 0.0, v_max))
            ppo_ms = (time.perf_counter() - t_ppo) * 1000.0
            ppo_times.append(ppo_ms)
            diag = None
            outcome_name = ""
            if spliner is None:
                selection_name = ""
            capped = False
            sf_delta, sf_v, filter_ms = rl_delta, rl_v, 0.0
            certify_ms = n_sims = math.nan
            if spliner is None:
                failed = ""
            if use_filter:
                t_f = time.perf_counter()
                vehicle_state = stack.vehicle_state(x, y, yaw, delta, v, r)
                rl_action = stack.action(rl_delta, rl_v)
                t_c = time.perf_counter()
                out = filt.certify(vehicle_state, tracked, rl_action, True)  # type:ignore
                certify_ms = (time.perf_counter() - t_c) * 1000.0
                filter_ms = perceive_ms + (time.perf_counter() - t_f) * 1000.0
                filter_times.append(filter_ms)
                sf_delta, sf_v = float(out.command.delta), float(out.command.v)
                outcome_name = out.outcome.name
                outcomes[outcome_name] += 1
                # The filter certified v_max_mps instead of the policy's speed:
                # outcome and selection describe the capped command.
                capped = bool(out.rl_speed_capped)
                if capped:
                    speed_capped += 1
                diag = out.diagnostics
                if diag is not None:
                    selection_name = diag.selection.name
                    selections[selection_name] += 1
                    n_sims = int(diag.rollouts)
                    failed = str(diag.failed_constraint)
            applied_delta, applied_v = (
                (sf_delta, sf_v) if sf == "on" else (rl_delta, rl_v)
            )
            if sf == "shadow":
                filt.set_published(stack.action(applied_delta, applied_v))  # type:ignore
            if use_filter and (
                abs(sf_delta - rl_delta) > INTERVENTION_EPS_STEER_RAD
                or abs(sf_v - rl_v) > INTERVENTION_EPS_SPEED_MPS
            ):
                interventions += 1
            predicted = [] if diag is None else list(diag.predicted_states)
            log.add(
                {
                    "step": steps,
                    "sim_time": sim_time,
                    "lap": len(lap_times),
                    "s": float(frenet.s),
                    "s_norm": float(frenet.s) / length if length else math.nan,
                    "n": float(frenet.n),
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "delta": delta,
                    "v": v,
                    "r": r,
                    "rl_delta": rl_delta,
                    "rl_v": rl_v,
                    "sf_delta": sf_delta,
                    "sf_v": sf_v,
                    "applied_delta": applied_delta,
                    "applied_v": applied_v,
                    "outcome": outcome_name,
                    "selection": selection_name,
                    "n_raw": len(raw_items),
                    **_rect_log(
                        "raw_", _nearest([it.rect for it in raw_items], x, y), False
                    ),
                    "n_tracked": len(tracked),
                    **_rect_log("obs_", _nearest(tracked, x, y), True),
                    **_constraint_log(diag),
                    "ppo_ms": ppo_ms,
                    "filter_ms": filter_ms,
                    "certify_ms": certify_ms,
                    "n_sims": n_sims,
                    "failed": failed,
                    "speed_capped": int(capped),
                    **_hidden_log(venv.vehicle_state(0), diag),
                }
            )
            _render(
                renderer, venv, tracked, outcome_name, predicted, x, y, dt, slowdown, t0
            )
            a_delta, a_v = observation.action_from_command(
                applied_delta, applied_v, steer_lock, v_max
            )
            applied_norm = np.array(
                [float(np.clip(a_delta, -1.0, 1.0)), float(np.clip(a_v, -1.0, 1.0))],
                dtype=np.float32,
            )
            delta_sum += np.abs(applied_norm - prev_action)
            prev_action = applied_norm
            venv.actions[0] = applied_norm
            venv._lib.rr_step(venv._vec)
            sim_time += dt
            steps += 1
            if v >= PARK_STOP_SPEED_MPS:
                has_moved = True
                standstill_s = 0.0
            else:
                standstill_s += dt
            if has_moved and standstill_s >= PARK_STANDSTILL_S:
                parked = True
                break
            cur_lap = int(venv._lib.rr_current_lap(venv._vec, 0))
            if cur_lap > prev_lap:
                prev_lap = cur_lap
                if not first_boundary:
                    first_boundary = True
                else:
                    lap_times.append(float(venv._lib.rr_last_lap_time(venv._vec, 0)))
                    if len(lap_times) >= num_laps:
                        break
            if venv.terminals[0]:
                kind = venv.done_kind(0)
                if kind == COLLISION:
                    crashed = True
                elif kind == COMPLETE:
                    completed = True
                    if len(lap_times) < num_laps:
                        lap_times.append(float(venv.done_lap_time(0)))
                break
    finally:
        venv.close()
        written = log.write()

    if completed or len(lap_times) >= num_laps:
        outcome = COMPLETED
    elif crashed:
        outcome = CRASHED
    elif parked:
        outcome = PARKED
    else:
        outcome = TIMEOUT
    denom = max(steps, 1)
    return LayoutResult(
        index=index,
        n_obstacles=len(layout),
        outcome=outcome,
        lap_times=lap_times,
        steps=steps,
        sim_time=sim_time,
        terminal_v=terminal_v,
        steer_rate=float(delta_sum[0] / denom),
        speed_cmd_rate=float(delta_sum[1] / denom),
        interventions=interventions,
        speed_capped=speed_capped,
        outcomes=dict(outcomes),
        selections=dict(selections),
        ppo_ms=timing_summary(ppo_times),
        filter_ms=timing_summary(filter_times) if use_filter else {},
        log_path=None if written is None else written.name,
    )
