"""Accuracy of the safety filter's rollout prediction against the plant and the car.

Two parts, each writing JSON (and plotting CSVs) under ``--out``:

``sim``
    Runs the eval tick loop (``evaluation.filter.run_layout``, filter on,
    detector obstacles, unchanged) with a recording wrapper around the filter
    and the plant library. Per tick it keeps the measured state handed to
    ``certify``, the plant's true state, the exported rollout
    (``predicted_states``, ``predicted_commands``), the hidden-state estimate
    and the applied action.

    * One period: stage 1 of the published rollout against the plant one
      control period later (true state, and the next measured state).
    * Horizon: at every ``--stride``-th tick the plant is cloned by
      deterministic replay (a fresh plant with the same seed and reset
      sequence, driven by the logged applied actions up to that tick; the
      clone's state is checked bit-for-bit against the recorded true state)
      and then driven open loop with the command sequence the published
      rollout assumed: the published command for stage 1 and
      ``predicted_commands[j]`` (the backup command the rollout computed on
      its own predicted state) for stage j+1. Per-stage position, heading
      and speed error of the rollout against the clone.
    * Hidden states: carried estimate and steady-state completion against
      plant truth, through ``observer_error.summarise``.

``summary``
    Merges the part JSONs into ``summary.json``.

    uv run --frozen python evaluation/tools/prediction_error.py sim --sim2real on --layouts 0-19
    uv run --frozen python evaluation/tools/prediction_error.py summary
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from evaluation.common import EVAL_SEED  # noqa: E402  (also puts sf/, sim/ on sys.path)
from evaluation.log import HIDDEN  # noqa: E402

OUT = REPO / "outputs" / "prediction"
RUN, SEED, CHECKPOINT = "obstacle-agent", 17, "best_eval"
KINEMATIC_FLOOR_MPS = 0.5  # sf_model.h kKinematicFloorMps
PERIOD_S = 0.067  # sf_tuning.h kControlPeriodS
# rr_vehicle_state order after the six measured states.
TRUE_HIDDEN = ("beta", "accel", "delta_dot", "fy_f", "fy_r")


def _wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _q(x: np.ndarray) -> dict[str, float]:
    """Median, p95, p99, mean, RMS and count of absolute errors."""
    x = np.abs(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "mean": float(np.mean(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
    }


# --------------------------------------------------------------------------- sim


class _Recorder:
    """Per-tick record filled by the filter and plant-library wrappers."""

    def __init__(self) -> None:
        self.venv = None
        self.meas: list[list[float]] = []  # measured state given to certify
        self.true: list[np.ndarray] = []  # plant state at certify time
        self.pred: list[np.ndarray] = []  # (S+1, 6) x y yaw delta v r
        self.cmds: list[np.ndarray] = []  # (S, 2) delta v, command space
        self.pub: list[tuple[float, float]] = []
        self.outcome: list[str] = []
        self.est: list[list[float]] = []
        self.ss: list[list[float]] = []
        self.actions: list[np.ndarray] = []  # normalised action stepped
        self.terminal: list[bool] = []
        self.true_after: list[np.ndarray] = []  # plant state after the step


class _FilterProxy:
    """SafetyFilter that records every certify call."""

    def __init__(self, filt, rec: _Recorder) -> None:
        self._f, self._rec = filt, rec

    def __getattr__(self, name):
        return getattr(self._f, name)

    def certify(self, state, obstacles, action, include_diagnostics=False):
        out = self._f.certify(state, obstacles, action, True)
        rec, diag = self._rec, out.diagnostics
        rec.meas.append([state.x, state.y, state.yaw, state.delta, state.v, state.r])
        rec.true.append(rec.venv.vehicle_state(0).astype(np.float64))
        rec.pred.append(np.array([[s.x, s.y, s.yaw, s.delta, s.v, s.r] for s in diag.predicted_states]))
        rec.cmds.append(np.array([[c.delta, c.v] for c in diag.predicted_commands]).reshape(-1, 2))
        rec.pub.append((float(out.command.delta), float(out.command.v)))
        rec.outcome.append(out.outcome.name)
        rec.est.append([float(getattr(diag.hidden_estimate, n)) for n in HIDDEN])
        rec.ss.append([float(getattr(diag.hidden_steady_state, n)) for n in HIDDEN])
        return out


class _LibProxy:
    """Plant library whose rr_step records the action and the resulting state."""

    def __init__(self, lib, venv, rec: _Recorder) -> None:
        self._lib, self._venv, self._rec = lib, venv, rec

    def __getattr__(self, name):
        return getattr(self._lib, name)

    def rr_step(self, vec):
        self._rec.actions.append(self._venv.actions[0].copy())
        self._lib.rr_step(vec)
        self._rec.terminal.append(bool(self._venv.terminals[0]))
        self._rec.true_after.append(self._venv.vehicle_state(0).astype(np.float64))


def _record_layout(cfg, agent, layout, index: int, sim2real: bool, laps: int) -> _Recorder:
    """Run the unchanged eval tick loop with recording wrappers patched in."""
    import evaluation.filter as ef

    rec = _Recorder()
    stack_fn, make_fn = ef._stack, ef.make_venv

    def stack_patch(*a, **k):
        track, det, trk, filt = stack_fn(*a, **k)
        return track, det, trk, _FilterProxy(filt, rec)

    def make_patch(*a, **k):
        venv = make_fn(*a, **k)
        rec.venv = venv
        venv._lib = _LibProxy(venv._lib, venv, rec)
        return venv

    ef._stack, ef.make_venv = stack_patch, make_patch
    try:
        ef.run_layout(agent, cfg, layout, index, sf="on", sim2real=sim2real,
                      obstacles="detector", num_laps=laps)
    finally:
        ef._stack, ef.make_venv = stack_fn, make_fn
    return rec


def _clone_horizon(cfg, layout, index: int, sim2real: bool, laps: int, rec: _Recorder,
                   ticks: list[int]) -> dict:
    """Clone the plant at each tick by replay and drive it with the rollout's commands."""
    import observation
    from ppo.layouts import to_sim
    from ppo.rollout import env_kwargs, episode_tick_cap, make_venv

    car = env_kwargs(cfg, sim2real)
    steer_lock, v_max = float(car["s_max"]), float(car["v_max"])
    m = len(ticks)
    stages = [len(rec.cmds[t]) for t in ticks]
    n_steps = max(t + s for t, s in zip(ticks, stages))
    acts = np.zeros((n_steps, m, 2), dtype=np.float32)
    main = np.asarray(rec.actions, dtype=np.float32)
    cmd_gap = np.full(m, np.nan)
    for j, (t, s) in enumerate(zip(ticks, stages)):
        acts[:t, j] = main[:t]
        acts[t, j] = main[t]  # the published command, as applied
        cmd_gap[j] = max(abs(rec.cmds[t][0, 0] - rec.pub[t][0]), abs(rec.cmds[t][0, 1] - rec.pub[t][1]))
        for k in range(1, s):
            a = observation.action_from_command(float(rec.cmds[t][k, 0]), float(rec.cmds[t][k, 1]),
                                                steer_lock, v_max)
            acts[t + k, j] = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
        acts[t + s:, j] = acts[t + s - 1, j]
    plant = [np.full((s + 1, 11), np.nan) for s in stages]
    alive = np.ones(m, dtype=bool)
    kinds = np.full(m, -1)  # plant done_kind when the clone ended inside the horizon
    replay_gap = np.full(m, np.nan)
    venv = make_venv(cfg, m, EVAL_SEED, sim2real=sim2real)
    try:
        for j in range(m):
            venv.seed(j, EVAL_SEED * 1000 + index)
            venv.set_lap_cap(j, laps + 1)
            venv.set_tick_cap(j, episode_tick_cap(cfg))
        venv.reset()
        for j in range(m):
            venv.reset_env(j)
            venv.set_obstacles(j, to_sim(layout))
        for i in range(n_steps):
            for j, (t, s) in enumerate(zip(ticks, stages)):
                if i == t:
                    st = venv.vehicle_state(j).astype(np.float64)
                    replay_gap[j] = float(np.max(np.abs(st - rec.true[t])))
                    plant[j][0] = st
            venv.actions[:] = acts[i]
            venv._lib.rr_step(venv._vec)
            for j, (t, s) in enumerate(zip(ticks, stages)):
                if t <= i < t + s and alive[j]:
                    if venv.terminals[j]:
                        alive[j] = False
                        kinds[j] = int(venv.done_kind(j))
                        continue
                    plant[j][i - t + 1] = venv.vehicle_state(j)
    finally:
        venv.close()
    return {"plant": plant, "replay_gap": replay_gap, "cmd_gap": cmd_gap, "kind": kinds}


def _sim_job(item: dict) -> dict:
    """One layout: record the tick loop, then one-period, horizon and hidden data."""
    import torch as T

    T.set_num_threads(1)
    from evaluation.cli import _cached_agent

    cfg, agent = _cached_agent(RUN, SEED, CHECKPOINT)
    layout, index, sim2real, laps = item["layout"], item["index"], item["sim2real"], item["laps"]
    rec = _record_layout(cfg, agent, layout, index, sim2real, laps)
    n = len(rec.meas)
    meas = np.asarray(rec.meas)
    true = np.asarray(rec.true)
    after = np.asarray(rec.true_after)
    term = np.asarray(rec.terminal, dtype=bool)
    outcome = np.asarray(rec.outcome)

    # One period: tick t's stage 1 vs the plant after step t (true) and the
    # measured state of tick t+1. Skip the first tick (no carried estimate,
    # u_prev = u_rl) and the terminal step (the plant resets inside it).
    t1 = np.arange(1, n - 1)
    t1 = t1[~term[t1] & (meas[t1, 4] >= KINEMATIC_FLOOR_MPS)]
    s1 = np.array([rec.pred[t][1] for t in t1]).reshape(-1, 6)
    one = {
        "pos_true": np.hypot(s1[:, 0] - after[t1, 0], s1[:, 1] - after[t1, 1]),
        "yaw_true": np.degrees(_wrap(s1[:, 2] - after[t1, 2])),
        "v_true": s1[:, 4] - after[t1, 4],
        "pos_meas": np.hypot(s1[:, 0] - meas[t1 + 1, 0], s1[:, 1] - meas[t1 + 1, 1]),
        "yaw_meas": np.degrees(_wrap(s1[:, 2] - meas[t1 + 1, 2])),
        "v_meas": s1[:, 4] - meas[t1 + 1, 4],
        # Displacement over the period, prediction vs plant: removes the
        # measurement offset of stage 0.
        "drift_pos_true": np.hypot((s1[:, 0] - meas[t1, 0]) - (after[t1, 0] - true[t1, 0]),
                                   (s1[:, 1] - meas[t1, 1]) - (after[t1, 1] - true[t1, 1])),
        "drift_yaw_true": np.degrees(_wrap((s1[:, 2] - meas[t1, 2]) - (after[t1, 2] - true[t1, 2]))),
        # Stage 0 = the measured state the rollout starts from.
        "pos0_true": np.hypot(meas[t1, 0] - true[t1, 0], meas[t1, 1] - true[t1, 1]),
        "v0_true": meas[t1, 4] - true[t1, 4],
        "outcome": outcome[t1],
        "v": meas[t1, 4],
        "r": meas[t1, 5],
    }

    # Horizon clones.
    stride = item["stride"]
    ticks = [int(t) for t in t1[::stride]]
    hor = {"tick": [], "stages": [], "err": [], "replay_gap": [], "cmd_gap": [], "outcome": [], "kind": []}
    example = None
    if ticks:
        cl = _clone_horizon(cfg, layout, index, sim2real, laps, rec, ticks)
        for j, t in enumerate(ticks):
            pred, pl = rec.pred[t], cl["plant"][j]
            s = len(rec.cmds[t])
            e = np.full((s + 1, 5), np.nan)
            p = pred[: s + 1]
            e[:, 0] = np.hypot(p[:, 0] - pl[:, 0], p[:, 1] - pl[:, 1])
            e[:, 1] = np.degrees(_wrap(p[:, 2] - pl[:, 2]))
            e[:, 2] = p[:, 4] - pl[:, 4]
            # Drift: displacement and heading change since stage 0, so the
            # measurement offset the rollout starts from is removed.
            e[:, 3] = np.hypot((p[:, 0] - p[0, 0]) - (pl[:, 0] - pl[0, 0]),
                               (p[:, 1] - p[0, 1]) - (pl[:, 1] - pl[0, 1]))
            e[:, 4] = np.degrees(_wrap((p[:, 2] - p[0, 2]) - (pl[:, 2] - pl[0, 2])))
            hor["tick"].append(t)
            hor["stages"].append(s)
            hor["err"].append(e)
            hor["replay_gap"].append(cl["replay_gap"][j])
            hor["cmd_gap"].append(cl["cmd_gap"][j])
            hor["outcome"].append(rec.outcome[t])
            hor["kind"].append(int(cl["kind"][j]))
        if item["keep_example"]:
            example = {
                "ticks": ticks,
                "pred": [rec.pred[t] for t in ticks],
                "cmds": [rec.cmds[t] for t in ticks],
                "plant": cl["plant"],
                "outcome": [rec.outcome[t] for t in ticks],
            }

    # Hidden states, same mask as observer_error.load.
    est, ss = np.asarray(rec.est), np.asarray(rec.ss)
    hmask = np.isfinite(est[:, 0]) & (meas[:, 4] >= KINEMATIC_FLOOR_MPS)
    hmask[:1] = False
    hidden = {"v": meas[hmask, 4], "r": meas[hmask, 5]}
    for i, name in enumerate(HIDDEN):
        hidden[f"est_{name}"] = est[hmask, i]
        hidden[f"ss_{name}"] = ss[hmask, i]
        hidden[f"true_{name}"] = true[hmask, 6 + TRUE_HIDDEN.index(name)]
    return {"index": index, "one": one, "hor": hor, "hidden": hidden, "example": example}


def _parse_layouts(spec: str, total: int) -> list[int]:
    """'all', 'a-b' (inclusive) or comma list."""
    if spec == "all":
        return list(range(total))
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def _horizon_stats(err: list[np.ndarray], stages: np.ndarray) -> list[dict]:
    """Per-stage position/heading/speed error stats across clone samples."""
    s_max = int(stages.max())
    rows = []
    for k in range(s_max + 1):
        col = np.array([e[k] for e in err if e.shape[0] > k])
        if col.size == 0:
            continue
        ok = np.isfinite(col[:, 0])
        rows.append({
            "stage": k,
            "t_s": k * PERIOD_S,
            "n": int(ok.sum()),
            "pos_m": _q(col[ok, 0]),
            "yaw_deg": _q(col[ok, 1]),
            "v_mps": _q(col[ok, 2]),
            "drift_pos_m": _q(col[ok, 3]),
            "drift_yaw_deg": _q(col[ok, 4]),
        })
    return rows


def cmd_sim(args) -> None:
    from evaluation.cli import _init_worker, _layouts

    layouts, indices = _layouts(args.eval_set, "all")
    chosen = _parse_layouts(args.layouts, len(layouts))
    tag = f"sim2real-{'on' if args.sim2real else 'off'}"
    out = args.out / "sim" / tag
    out.mkdir(parents=True, exist_ok=True)
    items = [
        {"layout": layouts[k], "index": indices[k], "sim2real": args.sim2real, "laps": args.laps,
         "stride": args.stride, "keep_example": indices[k] == args.example_layout}
        for k in chosen
    ]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx, initializer=_init_worker) as pool:
        futs = [pool.submit(_sim_job, it) for it in items]
        for done, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            print(f"[{tag}] {done}/{len(items)} k {res['index']}", flush=True)
    results.sort(key=lambda r: r["index"])

    one = {k: np.concatenate([r["one"][k] for r in results]) for k in results[0]["one"]}
    err = [e for r in results for e in r["hor"]["err"]]
    stages = np.array([s for r in results for s in r["hor"]["stages"]])
    replay_gap = np.array([g for r in results for g in r["hor"]["replay_gap"]])
    cmd_gap = np.array([g for r in results for g in r["hor"]["cmd_gap"]])
    hor_outcome = np.array([o for r in results for o in r["hor"]["outcome"]])
    hidden = {k: np.concatenate([r["hidden"][k] for r in results]) for k in results[0]["hidden"]}

    from evaluation.tools.observer_error import summarise

    final_err = np.array([e[-1, 0] for e in err])
    summary = {
        "condition": tag,
        "run": f"{RUN} seed {SEED} {CHECKPOINT}",
        "eval_set": args.eval_set,
        "layouts": [int(r["index"]) for r in results],
        "laps": args.laps,
        "one_period": {
            "ticks": int(one["pos_true"].size),
            "vs_true": {"pos_m": _q(one["pos_true"]), "yaw_deg": _q(one["yaw_true"]),
                        "v_mps": _q(one["v_true"]),
                        "v_bias_mps": float(np.mean(one["v_true"]))},
            "vs_measured": {"pos_m": _q(one["pos_meas"]), "yaw_deg": _q(one["yaw_meas"]),
                            "v_mps": _q(one["v_meas"])},
            "drift_vs_true": {"pos_m": _q(one["drift_pos_true"]), "yaw_deg": _q(one["drift_yaw_true"])},
            "stage0_measurement": {"pos_m": _q(one["pos0_true"]), "v_mps": _q(one["v0_true"])},
            "by_outcome_pos_true_m": {
                o: _q(one["pos_true"][one["outcome"] == o]) for o in sorted(set(one["outcome"]))
            },
        },
        "horizon": {
            "samples": int(len(err)),
            "stride_ticks": args.stride,
            "replay_max_state_gap": float(np.nanmax(replay_gap)) if replay_gap.size else None,
            "stage1_cmd_vs_published_max": float(np.nanmax(cmd_gap)) if cmd_gap.size else None,
            "stages_count": {int(s): int((stages == s).sum()) for s in np.unique(stages)},
            "truncated_by_terminal": int(sum(np.isnan(e[-1, 0]) for e in err)),
            # done_kind of clones that ended inside the horizon (see roboracer_sim COLLISION/COMPLETE).
            "terminal_kinds": {int(k): int(c) for k, c in zip(*np.unique(
                [k for r in results for k in r["hor"]["kind"] if k >= 0] or [-1], return_counts=True)) if k >= 0},
            "per_stage": _horizon_stats(err, stages),
            "per_stage_18": _horizon_stats([e for e, s in zip(err, stages) if s == 18], stages[stages == 18]) if (stages == 18).any() else [],
            "per_stage_24": _horizon_stats([e for e, s in zip(err, stages) if s == 24], stages[stages == 24]) if (stages == 24).any() else [],
            "final_stage_pos_m_by_outcome": {
                o: _q(final_err[hor_outcome == o]) for o in sorted(set(hor_outcome))
            },
        },
        "hidden": summarise(hidden),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # Per-tick one-period errors and per-sample horizon errors for plotting.
    with (out / "one_period.csv").open("w") as f:
        keys = list(one)
        f.write(",".join(keys) + "\n")
        for i in range(one["pos_true"].size):
            f.write(",".join(str(one[k][i]) if k == "outcome" else f"{one[k][i]:.6g}" for k in keys) + "\n")
    with (out / "horizon_errors.csv").open("w") as f:
        f.write("sample,stage,t_s,pos_m,yaw_deg,v_mps,drift_pos_m,drift_yaw_deg\n")
        for j, e in enumerate(err):
            for k in range(e.shape[0]):
                f.write(f"{j},{k},{k * PERIOD_S:.3f}," + ",".join(f"{x:.6g}" for x in e[k]) + "\n")
    _write_example(results, out, final_err, stages)
    print(json.dumps({"one_period": summary["one_period"]["vs_true"],
                      "final_stage": summary["horizon"]["per_stage"][-1]}, indent=1, default=float))


def _write_example(results, out: Path, final_err: np.ndarray, stages: np.ndarray) -> None:
    """Example trajectory pair: the sample in the example layout with the most
    stages whose final position error is closest to the condition's median."""
    owner = next((r for r in results if r["example"]), None)
    if owner is None:
        return
    ex = owner["example"]
    if ex is None:
        return
    s_max = max(len(c) for c in ex["cmds"])
    target = float(np.nanmedian(final_err[stages == s_max])) if (stages == s_max).any() else float(np.nanmedian(final_err))
    best, best_d = None, math.inf
    for j, (pred, plant) in enumerate(zip(ex["pred"], ex["plant"])):
        if len(ex["cmds"][j]) != s_max or not np.isfinite(plant[-1, 0]):
            continue
        d = abs(math.hypot(pred[-1, 0] - plant[-1, 0], pred[-1, 1] - plant[-1, 1]) - target)
        if d < best_d:
            best, best_d = j, d
    if best is None:
        return
    pred, plant, cmds = ex["pred"][best], ex["plant"][best], ex["cmds"][best]
    with (out / "example_trajectory.csv").open("w") as f:
        f.write(f"# layout {owner['index']} tick {ex['ticks'][best]} outcome {ex['outcome'][best]}; median final error target {target:.4f} m\n")
        f.write("stage,t_s,pred_x,pred_y,pred_yaw,pred_v,plant_x,plant_y,plant_yaw,plant_v,cmd_delta,cmd_v\n")
        for k in range(pred.shape[0]):
            cd, cv = (cmds[k] if k < len(cmds) else (math.nan, math.nan))
            f.write(f"{k},{k * PERIOD_S:.3f},{pred[k, 0]:.5f},{pred[k, 1]:.5f},{pred[k, 2]:.5f},{pred[k, 4]:.5f},"
                    f"{plant[k, 0]:.5f},{plant[k, 1]:.5f},{plant[k, 2]:.5f},{plant[k, 4]:.5f},{cd:.5f},{cv:.5f}\n")


# --------------------------------------------------------------------------- summary


def cmd_summary(args) -> None:
    out: dict = {}
    for part in sorted((args.out / "sim").glob("*/summary.json")):
        out.setdefault("sim", {})[part.parent.name] = json.loads(part.read_text())
    for part in sorted((args.out / "hidden").glob("observer_*.json")):
        out.setdefault("hidden_existing_runs", {})[part.stem] = json.loads(part.read_text())
    (args.out / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out / 'summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--sim2real", choices=("on", "off"), required=True)
    s.add_argument("--eval-set", default="obs-set-dev-500")
    s.add_argument("--layouts", default="0-19", help="'all', 'a-b' or comma list")
    s.add_argument("--laps", type=int, default=2)
    s.add_argument("--stride", type=int, default=4, help="clone the plant every N ticks")
    s.add_argument("--example-layout", type=int, default=0)
    s.add_argument("--jobs", type=int, default=8)
    sub.add_parser("summary")
    args = p.parse_args()
    if args.cmd == "sim":
        args.sim2real = args.sim2real == "on"
        cmd_sim(args)
    else:
        cmd_summary(args)


if __name__ == "__main__":
    main()
