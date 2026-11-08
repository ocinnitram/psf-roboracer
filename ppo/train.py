"""Train a PPO racing policy on the C plant."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
import torch as T
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
T.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))

from evaluation.common import EVAL_SEED
from ppo.agent import PPOAgent
from ppo.config import (
    SEEDS,
    Config,
    TrainingConfig,
    dump,
    load,
    parse_set,
    resolve,
    score,
)
from ppo.layouts import Layout, Track, Zones, load_set, sample_layout, set_map, to_sim
from ppo.rollout import COMPLETE, episode_tick_cap, evaluate, make_venv, observation_size

REPO = Path(__file__).resolve().parents[1]
N_PARALLEL_ENV = 32
N_EMPTY_ENVS = 8
AVG_SCORE_NUM = 100
N_ACTIONS = 2


@dataclass
class PhaseState:
    """Phase-2 layout bookkeeping for the parallel envs."""

    n_obstacles: int = 0
    segment_start_step: int = 0
    segment_length: int = 1


def _set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch."""
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seeds set to {seed}")


def _is_better_eval(
    candidate: tuple[float, float], best: tuple[float, float] | None
) -> bool:
    """True when candidate beats best under score(), or no best is stored yet."""
    if best is None:
        return True
    return score(candidate[0], candidate[1]) > score(best[0], best[1])


def _n_empty_envs(cfg: Config) -> int:
    """How many workers stay on the empty track in phase 2."""
    if cfg.training.n_empty_envs is not None:
        return int(cfg.training.n_empty_envs)
    return N_EMPTY_ENVS


def _decay_parameters(
    total_steps: int,
    alpha: float,
    initial_c2: float,
    final_c2: float,
    agent: PPOAgent,
    global_step_cnt: int,
    segment_start_step: int,
    segment_length: int,
    caps_scale_hold: float | None,
) -> None:
    """Decay learning rate, entropy bonus, and CAPS weights."""
    lr_fraction = 1.0 - (global_step_cnt / max(total_steps, 1))
    new_lr = alpha * lr_fraction
    agent.actor.optimizer.param_groups[0]["lr"] = new_lr
    agent.critic.optimizer.param_groups[0]["lr"] = new_lr
    agent.set_caps_scale(lr_fraction if caps_scale_hold is None else caps_scale_hold)
    steps_in_segment = global_step_cnt - segment_start_step
    c2_fraction = 1.0 - min(steps_in_segment / max(segment_length, 1), 1.0)
    agent.set_c2(initial_c2 * c2_fraction + final_c2 * (1.0 - c2_fraction))


def _zones(cfg: Config) -> Zones | None:
    """Load placement zones when a zones file is configured."""
    if not cfg.obstacles.zones_file:
        return None
    return Zones.load(cfg.track.map)


def _eval_layouts(cfg: Config) -> list[Layout]:
    """Held-out layouts for in-loop evaluation, or one empty layout."""
    if cfg.training.eval_set:
        got = set_map(cfg.training.eval_set)
        if got != cfg.track.map:
            raise RuntimeError(
                f"eval set {cfg.training.eval_set!r} is map {got!r}, "
                f"track.map is {cfg.track.map!r}"
            )
        layouts = load_set(cfg.training.eval_set)
        if not layouts:
            raise RuntimeError(f"eval set {cfg.training.eval_set!r} is empty")
        return layouts
    return [()]


def _assign_layout(
    venv: Any,
    cfg: Config,
    track: Track,
    zones: Zones | None,
    env_i: int,
    rng: np.random.Generator,
) -> None:
    """Sample a layout onto one env after a reset."""
    layout = sample_layout(track, cfg.obstacles, zones, rng)
    venv.reset_env(env_i)
    venv.set_obstacles(env_i, to_sim(layout))


def _deploy_phase2(
    venv: Any,
    agent: PPOAgent,
    cfg: Config,
    track: Track,
    zones: Zones | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Install a sampled layout on each obstacle env."""
    n_obs = N_PARALLEL_ENV - _n_empty_envs(cfg)
    print(f"Phase 2: deploying {n_obs} obstacle layouts")
    for i in range(n_obs):
        _assign_layout(venv, cfg, track, zones, i, rng)
    agent.rollout_buffer.clear_data()
    return venv.obs.copy()


def _maybe_switch_finished_layouts(
    phase: PhaseState,
    venv: Any,
    cfg: Config,
    track: Track,
    zones: Zones | None,
    rng: np.random.Generator,
    dones: list[bool],
    infos: list[dict[str, Any]],
    states: np.ndarray,
) -> np.ndarray:
    """Resample an env's layout when it finishes, per training.layout_switch."""
    if phase.n_obstacles == 0 or cfg.training.layout_switch == "never":
        return states
    n_obs = N_PARALLEL_ENV - _n_empty_envs(cfg)
    on_crash = cfg.training.layout_switch == "episode"
    for i in range(n_obs):
        if not dones[i] or not (on_crash or infos[i].get("complete")):
            continue
        _assign_layout(venv, cfg, track, zones, i, rng)
        states[i] = venv.obs[i]
    return states


def _collect_step(
    states: np.ndarray,
    agent: PPOAgent,
    venv: Any,
) -> tuple[np.ndarray, list[bool], list[dict[str, Any]], np.ndarray]:
    """Sample actions, step the plant, and store the interleaved transitions."""
    actions_np, log_prob_tensors, val_tensors = agent.choose_action_batched(states)
    next_states, rewards, terminals = venv.step(actions_np)
    dones = [bool(flag) for flag in terminals]
    infos: list[dict[str, Any]] = []
    for i in range(N_PARALLEL_ENV):
        agent.store_transition(
            states[i],
            actions_np[i],
            log_prob_tensors[i],
            float(val_tensors[i].item()),
            float(rewards[i]),
            dones[i],
        )
        if not dones[i]:
            infos.append({})
            continue
        complete = venv.done_kind(i) == COMPLETE
        infos.append(
            {
                "complete": complete,
                "flying_lap_time": float(venv.done_lap_time(i)) if complete else None,
            }
        )
    return next_states, dones, infos, actions_np


def _log_episode_metrics(
    dones: list[bool],
    infos: list[dict[str, Any]],
    score_history: list[float],
    cfg: Config,
) -> float | None:
    """Record pace from completed flying laps. Returns the last lap time."""
    last_lap: float | None = None
    for done, info in zip(dones, infos):
        if not done:
            continue
        lap_time = info.get("flying_lap_time")
        score_history.append(
            float(cfg.track.opt_time) / float(lap_time) if lap_time else 0.0
        )
        if lap_time:
            last_lap = float(lap_time)
    return last_lap


def _setup_agent_folder(cfg: Config, seed: int) -> Path:
    """Create the run directory, refuse overwrites, and write the effective config."""
    agent_dir = REPO / "outputs" / "trained_agents" / cfg.training.run_name
    actor_file = agent_dir / f"_actor_seed_{seed}"
    critic_file = agent_dir / f"_critic_seed_{seed}"
    existing = [path for path in (actor_file, critic_file) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"checkpoint(s) already exist for run {cfg.training.run_name!r} seed {seed}: {names}"
        )
    agent_dir.mkdir(parents=True, exist_ok=True)
    config_dst = agent_dir / "config.yaml"
    if not config_dst.exists():
        config_dst.write_text(dump(cfg))
    return agent_dir


def _parse_warmstart(spec: str) -> tuple[str, str]:
    """Split run[:suffix]; default suffix is _best_eval."""
    if ":" in spec:
        run_name, suffix = spec.split(":", 1)
        return run_name, suffix
    return spec, "_best_eval"


def warmstart_overrides(warmstart: str | None) -> dict[str, Any]:
    """Config keys a warm-started run must inherit from its source: the architecture."""
    if not warmstart:
        return {}
    src_run, _ = _parse_warmstart(warmstart)
    src_cfg = REPO / "outputs" / "trained_agents" / src_run / "config.yaml"
    if not src_cfg.exists():
        raise RuntimeError(f"warmstart source has no config: {src_cfg}")
    with src_cfg.open() as handle:
        data = yaml.safe_load(handle)
    return {"ppo.hidden_size": int(data["ppo"]["hidden_size"])}


def _make_agent(cfg: Config, seed: int) -> PPOAgent:
    """Construct the actor-critic for this run and seed."""
    return PPOAgent(
        n_actions=N_ACTIONS,
        input_dims=observation_size(),
        gamma=cfg.ppo.gamma,
        alpha=cfg.ppo.alpha,
        gae_lambda=cfg.ppo.gae_lambda,
        policy_clip=cfg.ppo.clip,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        log_std_init=cfg.ppo.log_std_i,
        initial_c2=cfg.ppo.c2_i,
        normalise_advantages=cfg.ppo.normalise_advantages,
        target_kl=cfg.ppo.target_kl,
        c1=cfg.ppo.c1,
        base_path=REPO / "outputs" / "trained_agents",
        run_name=cfg.training.run_name,
        seed=seed,
        hidden_size=cfg.ppo.hidden_size,
        caps_lambda_t=cfg.ppo.caps_lambda_t,
        caps_lambda_s=cfg.ppo.caps_lambda_s,
        caps_sigma_s=cfg.ppo.caps_sigma_s,
        caps_lambda_t_v=cfg.ppo.caps_lambda_t_v,
        caps_lambda_s_v=cfg.ppo.caps_lambda_s_v,
    )


def _maybe_load_warmstart(agent: PPOAgent, warmstart: str | None, seed: int) -> None:
    """Copy a previous run's weights into this agent and load them."""
    if not warmstart:
        return
    src_run, suffix = _parse_warmstart(warmstart)
    src_dir = REPO / "outputs" / "trained_agents" / src_run
    src_actor = src_dir / f"_actor_seed_{seed}{suffix}"
    src_critic = src_dir / f"_critic_seed_{seed}{suffix}"
    missing = [str(path) for path in (src_actor, src_critic) if not path.exists()]
    if missing:
        raise RuntimeError(f"warmstart source missing: {', '.join(missing)}")
    shutil.copy(src_actor, agent.actor.path)
    shutil.copy(src_critic, agent.critic.path)
    agent.load_models()
    print(f"Warm-started from {src_run}{suffix}")


def _wandb_log(wandb_run: Any, payload: dict[str, Any], step: int) -> None:
    """Log a dict to wandb when a run is active."""
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def run(
    cfg: Config,
    seed: int,
    warmstart: str | None,
    wandb_run: Any | None = None,
) -> tuple[float, float]:
    """Train one seed and return the best (success_rate, pace)."""
    print(f"Starting training with SEED: {seed}")
    _set_seeds(seed)
    _setup_agent_folder(cfg, seed)
    agent = _make_agent(cfg, seed)
    _maybe_load_warmstart(agent, warmstart, seed)

    phase2 = cfg.training.n_phase2_steps > 0
    total_steps = cfg.training.n_phase1_steps + cfg.training.n_phase2_steps
    eval_interval = cfg.training.eval_interval
    patience_evals = cfg.training.patience_evals
    ckpt_interval = cfg.training.ckpt_interval
    track = Track.load(cfg.track.map)
    zones = _zones(cfg) if phase2 else None
    rng = np.random.default_rng(seed)
    eval_layouts = _eval_layouts(cfg)

    if phase2:
        n_empty = _n_empty_envs(cfg)
        print(
            f"Phase 2: {n_empty} empty + {N_PARALLEL_ENV - n_empty} obstacle envs"
        )
        phase = PhaseState(
            n_obstacles=0,
            segment_length=(
                cfg.training.n_phase2_steps
                if cfg.training.n_phase1_steps == 0
                else cfg.training.n_phase1_steps
            ),
        )
    else:
        print(f"Empty-track training: {total_steps} steps")
        phase = PhaseState(n_obstacles=0, segment_length=total_steps)

    venv = make_venv(cfg, N_PARALLEL_ENV, seed)
    lap_cap = int(cfg.training.num_laps) + 1
    tick_cap = episode_tick_cap(cfg)
    for i in range(N_PARALLEL_ENV):
        venv.set_lap_cap(i, lap_cap)
        venv.set_tick_cap(i, tick_cap)
    states = venv.reset()

    score_history: list[float] = []
    global_step_cnt = 0
    start_time = time.time()
    phase2_entered = False
    best_eval: tuple[float, float] | None = None
    evals_since_improve = 0
    next_eval_step = eval_interval
    next_ckpt_step = ckpt_interval
    last_eval: dict[str, float] | None = None
    last_lap: float | None = None
    last_update_time = start_time
    last_update_steps = 0
    collect_s = 0.0
    stop_reason: str | None = None

    try:
        print(f"Plant: {N_PARALLEL_ENV} envs")
        while global_step_cnt < total_steps:
            _decay_parameters(
                total_steps,
                cfg.ppo.alpha,
                cfg.ppo.c2_i,
                cfg.ppo.c2_f,
                agent,
                global_step_cnt,
                phase.segment_start_step,
                phase.segment_length,
                cfg.ppo.caps_scale_hold,
            )
            t0 = time.perf_counter()
            next_states, dones, infos, _actions = _collect_step(states, agent, venv)
            collect_s += time.perf_counter() - t0
            global_step_cnt += N_PARALLEL_ENV

            if len(agent.rollout_buffer.rewards) >= cfg.ppo.rollout:
                agent.store_rollout_buffer()
                t0 = time.perf_counter()
                diag = agent.learn(next_states, dones, N_PARALLEL_ENV)
                update_s = time.perf_counter() - t0
                if not np.isfinite(diag["actor_loss"]) or not np.isfinite(
                    diag["critic_loss"]
                ):
                    stop_reason = "non-finite loss"
                    break
                now = time.time()
                sps = (global_step_cnt - last_update_steps) / max(
                    now - last_update_time, 1e-9
                )
                last_update_time, last_update_steps = now, global_step_cnt
                avg_score = (
                    float(np.mean(score_history[-AVG_SCORE_NUM:]))
                    if score_history
                    else 0.0
                )
                _wandb_log(
                    wandb_run,
                    {
                        "Losses/Actor": diag["actor_loss"],
                        "Losses/Critic": diag["critic_loss"],
                        "Losses/Entropy": diag["entropy"],
                        "Losses/ApproxKL": diag["approx_kl"],
                        "Schedule/LR": agent.actor.optimizer.param_groups[0]["lr"],
                        "Schedule/Entropy": agent.c2,
                        "Schedule/LogStd": float(
                            agent.actor.log_std.detach().mean().item()
                        ),
                        "System/SPS": sps,
                        "Training/Score": avg_score,
                        **(
                            {"Training/LapTime": last_lap}
                            if last_lap is not None
                            else {}
                        ),
                    },
                    global_step_cnt,
                )
                lap_str = f"{last_lap:.2f}s" if last_lap is not None else "NA"
                eval_str = (
                    f"{last_eval['success_rate']:.2f}" if last_eval else "NA"
                )
                print(
                    f"Time:{(now - start_time) / 60.0:6.2f}m | "
                    f"Steps:{global_step_cnt / 1.0e6:6.2f}M | SPS:{sps:6.0f} | "
                    f"col:{collect_s:4.1f}s upd:{update_s:4.1f}s | "
                    f"Score:{avg_score:6.3f} | lap:{lap_str} | "
                    f"KL:{diag['approx_kl']:.4f} | Ent:{diag['entropy']:6.3f} | "
                    f"Eval:{eval_str}"
                )
                collect_s = 0.0

            lap = _log_episode_metrics(dones, infos, score_history, cfg)
            if lap is not None:
                last_lap = lap
            states = _maybe_switch_finished_layouts(
                phase, venv, cfg, track, zones, rng, dones, infos, next_states
            )

            if global_step_cnt >= next_ckpt_step:
                agent.save_models_with_suffix(f"_ckpt_{global_step_cnt}")
                next_ckpt_step += ckpt_interval

            if (
                phase2
                and not phase2_entered
                and global_step_cnt >= cfg.training.n_phase1_steps
            ):
                phase2_entered = True
                if cfg.training.n_phase1_steps > 0:
                    print(f"Phase 1 done at {global_step_cnt} steps")
                    agent.save_models_with_suffix("_phase1")
                phase.n_obstacles = int(cfg.obstacles.n_obstacles)
                states = _deploy_phase2(venv, agent, cfg, track, zones, rng)
                phase.segment_start_step = global_step_cnt
                phase.segment_length = cfg.training.n_phase2_steps

            if global_step_cnt >= next_eval_step:
                em = evaluate(agent, cfg, eval_layouts, EVAL_SEED)
                last_eval = em
                _wandb_log(
                    wandb_run,
                    {
                        "Eval/SuccessRate": em["success_rate"],
                        "Eval/LapTime": em["mean_lap_time"],
                        "Eval/Pace": em["pace"],
                        "Eval/SteerRate": em["steer_rate"],
                        "Eval/SpeedCmdRate": em["speed_cmd_rate"],
                        # The sweep objective; wandb early termination reads it by name.
                        "score": score(em["success_rate"], em["pace"]),
                    },
                    global_step_cnt,
                )
                lap_str = (
                    f"{em['mean_lap_time']:.2f}s"
                    if not np.isnan(em["mean_lap_time"])
                    else "NA"
                )
                print(
                    f"EVAL @{global_step_cnt / 1.0e6:.2f}M | "
                    f"success:{em['success_rate']:.2f} | pace:{em['pace']:.3f} | "
                    f"lap:{lap_str}"
                )
                metric = (em["success_rate"], em["pace"])
                if _is_better_eval(metric, best_eval):
                    best_eval = metric
                    evals_since_improve = 0
                    agent.save_models_with_suffix("_best_eval")
                else:
                    evals_since_improve += 1
                    if evals_since_improve >= patience_evals:
                        stop_reason = (
                            f"no eval improvement for {patience_evals} evals"
                        )
                        break
                next_eval_step += eval_interval

        if best_eval is None:
            em = evaluate(agent, cfg, eval_layouts, EVAL_SEED)
            last_eval = em
            best_eval = (em["success_rate"], em["pace"])
            agent.save_models_with_suffix("_best_eval")
            print(
                f"EVAL @end | success:{em['success_rate']:.2f} | pace:{em['pace']:.3f}"
            )

        if stop_reason:
            print(f"Stopped early at {global_step_cnt} steps: {stop_reason}")
        agent.save_models()
    finally:
        venv.close()

    print(f"Training with SEED {seed} completed successfully!")
    return best_eval if best_eval is not None else (0.0, 0.0)


def _parse_seeds(spec: str | None) -> list[int]:
    """Parse a comma-separated seed list, or the first seed of the table."""
    if spec is None:
        return [SEEDS[0]]
    return [int(part) for part in spec.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ./train [phase|path|run] [--seeds] [--layout-switch] [--warmstart] [--base] [--set] [--wandb] [--tag]."""
    parser = argparse.ArgumentParser(description="Train a PPO racing policy")
    parser.add_argument(
        "target",
        nargs="?",
        default="empty",
        help="Phase name (empty, obstacles), config path, or trained-run name (default empty)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=f"Comma-separated seeds (default {SEEDS[0]}; the thesis tables use "
        f"{','.join(str(s) for s in SEEDS)})",
    )
    parser.add_argument(
        "--layout-switch",
        choices=get_args(TrainingConfig.model_fields["layout_switch"].annotation),
        default=None,
        help="When an obstacle env draws a new layout; a value other than the config's "
        "appends -<value> to the run name",
    )
    parser.add_argument(
        "--warmstart",
        default=None,
        help="Load weights from run[:suffix] (default suffix _best_eval)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Load this config instead of target; target becomes training.run_name",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="Dotted override, e.g. training.n_phase1_steps=40000",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb project ppo-lab (off by default)",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Wandb tag for every seed run of this call, e.g. seeds-<sweep-id>; repeatable",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    overrides = {**parse_set(args.sets), **warmstart_overrides(args.warmstart)}
    source = args.base if args.base else args.target
    cfg = load(resolve(source), overrides)
    if args.base:
        cfg.training.run_name = args.target
    if args.layout_switch and args.layout_switch != cfg.training.layout_switch:
        cfg.training.layout_switch = args.layout_switch
        cfg.training.run_name = f"{cfg.training.run_name}-{args.layout_switch}"
    print(dump(cfg), flush=True)

    seeds = _parse_seeds(args.seeds)
    for seed in seeds:
        wandb_run: Any | None = None
        if args.wandb:
            import wandb

            wandb_run = wandb.init(
                project="ppo-lab",
                group=cfg.training.run_name,
                job_type=f"seed_{seed}",
                tags=args.tags,
                config=cfg.model_dump(),
                dir=str(REPO / "outputs"),
            )
        try:
            run(cfg, seed, args.warmstart, wandb_run)
        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
