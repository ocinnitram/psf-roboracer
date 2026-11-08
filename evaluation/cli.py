"""Evaluate a trained run, with or without the safety filter."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "4")

import torch as T

T.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))

from evaluation.common import OUT_DIR, REPO, LayoutResult  # noqa: E402
from ppo.agent import PPOAgent  # noqa: E402
from ppo.config import Config, load, resolve  # noqa: E402
from ppo.layouts import Layout, load_set, set_map  # noqa: E402

_AGENTS: dict[tuple[str, int, str], tuple[Config, PPOAgent]] = {}
# Modules whose absence means sf/ is not built, not a bug.
_FILTER_MODULES = frozenset({"evaluation.filter", "ppo_sf_roboracer", "stack"})


def _checkpoint_suffix(checkpoint: str) -> str:
    """Map a checkpoint name to the actor-file suffix."""
    if checkpoint == "best_eval":
        return "_best_eval"
    if checkpoint == "final":
        return ""
    if checkpoint.isdigit():
        return f"_ckpt_{checkpoint}"
    raise ValueError(
        f"checkpoint must be best_eval, final, or a step number, got {checkpoint!r}"
    )


def _discover_seeds(run_dir: Path, suffix: str) -> list[int]:
    """Seeds that have an actor file with exactly this checkpoint suffix."""
    prefix = "_actor_seed_"
    seeds: list[int] = []
    for path in run_dir.glob(f"{prefix}*{suffix}"):
        rest = path.name[len(prefix) :]
        if suffix:
            if not rest.endswith(suffix):
                continue
            rest = rest[: -len(suffix)]
        if rest.isdigit():
            seeds.append(int(rest))
    return sorted(seeds)


def _parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated seed list."""
    seeds = [int(part) for part in spec.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds needs at least one seed")
    return seeds


def _parse_layout(spec: str) -> str | int:
    """Accept 'all', an index, or 'k<index>'."""
    text = spec.strip()
    if text.lower() == "all":
        return "all"
    if text[:1] in "kK" and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    raise argparse.ArgumentTypeError(f"layout must be all or an index, got {spec!r}")


def _on_off(spec: str) -> bool:
    """Parse an on/off flag."""
    if spec in ("on", "off"):
        return spec == "on"
    raise argparse.ArgumentTypeError(f"expected on or off, got {spec!r}")


def _default_jobs() -> int:
    """Half the CPUs, capped at 8."""
    return max(1, min(8, (os.cpu_count() or 1) // 2))


def load_agent(run: str, seed: int, checkpoint: str) -> tuple[Config, PPOAgent]:
    """Load config and actor weights for one run seed."""
    from ppo.train import _make_agent

    cfg = load(resolve(run))
    run_dir = REPO / "outputs" / "trained_agents" / run
    path = run_dir / f"_actor_seed_{seed}{_checkpoint_suffix(checkpoint)}"
    if not path.is_file():
        raise FileNotFoundError(
            f"no actor for run {run!r} seed {seed} checkpoint {checkpoint!r}: {path}"
        )
    agent = _make_agent(cfg, seed)
    agent.actor.updateActorPath(path)
    agent.actor.load_checkpoint()
    agent.actor.eval()
    return cfg, agent


def on_map(run: str, cfg: Config, map_name: str | None) -> Config:
    """The run's config moved to `map_name` (unchanged when None or its own map).

    Track numbers come from the evaluated map's track.yaml: length (s_norm),
    opt_time (tick cap, pace), start pose, and filter_v_max_mps (the filter is
    configured per map). The plant speed cap stays the one the policy trained
    with, since it scales the policy's speed action and speed observation.
    """
    if map_name is None or map_name == cfg.track.map:
        return cfg
    from ppo.rollout import env_kwargs

    moved = load(resolve(run), {"track.map": map_name})
    moved.track.max_speed_mps = float(env_kwargs(cfg)["v_max"])
    return moved


def _cached_agent(run: str, seed: int, checkpoint: str) -> tuple[Config, PPOAgent]:
    """Return a process-local cached agent for this seed and checkpoint."""
    key = (run, seed, checkpoint)
    if key not in _AGENTS:
        _AGENTS[key] = load_agent(run, seed, checkpoint)
    return _AGENTS[key]


def _init_worker() -> None:
    """Pin worker BLAS and Torch to one thread."""
    os.environ["TORCH_NUM_THREADS"] = "1"
    T.set_num_threads(1)


def _filter_job(item: dict[str, Any]) -> tuple[int, LayoutResult]:
    """Run one layout through the tick loop and return (seed, result)."""
    from evaluation.filter import run_layout

    if item["driver"] == "policy":
        cfg, agent = _cached_agent(item["run"], item["seed"], item["checkpoint"])
    else:
        cfg, agent = load(resolve(item["run"])), None
    log_path = Path(item["log_path"]) if item["log_path"] else None
    result = run_layout(
        agent,
        on_map(item["run"], cfg, item["map"]),
        item["layout"],
        item["index"],
        sf=item["sf"],
        sim2real=item["sim2real"],
        obstacles=item["obstacles"],
        num_laps=item["num_laps"],
        log_path=log_path,
        render=item["render"],
        slowdown=item["slowdown"],
        driver=item["driver"],
        speed_scale=item["speed_scale"],
    )
    return item["seed"], result


def _progress(seed: int, result: LayoutResult) -> None:
    """Print one finished-layout line."""
    laps = ",".join(f"{t:.2f}" for t in result.lap_times) if result.lap_times else "-"
    print(
        f"seed {seed} k {result.index} {result.outcome} laps {laps} "
        f"interventions {result.interventions}",
        flush=True,
    )


def _layouts(eval_set: str, layout: str | int) -> tuple[list[Layout], list[int]]:
    """Load the eval set and keep all layouts or one index."""
    if eval_set == "empty":
        entries: list[tuple[int, Layout]] = [(0, ())]
    else:
        loaded = load_set(eval_set)
        if not loaded:
            raise SystemExit(f"eval set {eval_set!r} has no layouts")
        entries = list(enumerate(loaded))
    if layout != "all":
        index = int(layout)
        match = [item for item in entries if item[0] == index]
        if not match:
            raise SystemExit(f"layout {index} is not in eval set {eval_set!r}")
        entries = match
    return [item[1] for item in entries], [item[0] for item in entries]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the eval command line."""
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Evaluate a trained run, with or without the safety filter",
    )
    parser.add_argument(
        "run",
        nargs="?",
        default="racing-agent",
        help="Trained-run name under outputs/trained_agents/ (default racing-agent)",
    )
    parser.add_argument(
        "--sf",
        choices=("on", "shadow", "off"),
        default="on",
        help="Apply the filter, run it in shadow, or apply the policy only",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=None,
        help="Comma-separated seeds (default: every seed with this checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_eval",
        help="best_eval, final, or a training step (default best_eval)",
    )
    parser.add_argument(
        "--eval-set",
        dest="eval_set",
        default=None,
        help="Layout set name, or empty (default: the run's training.eval_set or empty)",
    )
    parser.add_argument(
        "--map",
        dest="map",
        default=None,
        help=(
            "Evaluate on this map instead of the run's training map (cross-map): "
            "track numbers and filter v_max from its track.yaml, the policy's own "
            "plant speed cap; eval set defaults to empty and must be on this map"
        ),
    )
    parser.add_argument(
        "--layout",
        type=_parse_layout,
        default="all",
        help="Layout index or all (default all)",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=None,
        help="Flying laps (default: the run's training.num_laps)",
    )
    parser.add_argument(
        "--sim2real",
        type=_on_off,
        default=None,
        help="on or off (default: the run's training.sim2real)",
    )
    parser.add_argument(
        "--driver",
        choices=("policy", "ftg", "disparity", "spliner", "spliner-pp"),
        default="policy",
        help=(
            "Who drives: the run's policy, a reactive baseline on the scan "
            "(ftg: Evans' follow-the-gap, disparity: the disparity extender), or "
            "spliner (ForzaETH's evasion splines on the tracked obstacles, MAP controller; "
            "spliner-pp: the same with pure pursuit); tick loop, the run only supplies map, "
            "eval set and plant"
        ),
    )
    parser.add_argument(
        "--speed-scale",
        dest="speed_scale",
        type=float,
        default=1.0,
        help="Scale on a non-policy driver's speeds (default 1.0)",
    )
    parser.add_argument(
        "--obstacles",
        choices=("detector", "truth"),
        default="detector",
        help="Tracked detections or plant truth (tick loop only)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel layouts in the tick loop (default min(8, nproc/2); 1 with --render)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write filter_log_seed<S>_k<K>.csv in the tick loop",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Live window; forces the tick loop and jobs=1",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Pace wall time to slowdown * dt per tick when rendering",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if args.laps is not None and args.laps < 1:
        parser.error("--laps must be >= 1")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.slowdown <= 0:
        parser.error("--slowdown must be > 0")
    if args.speed_scale <= 0:
        parser.error("--speed-scale must be > 0")
    try:
        _checkpoint_suffix(args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _run_policy(
    run: str,
    seeds: Sequence[int],
    checkpoint: str,
    layouts: Sequence[Layout],
    indices: Sequence[int],
    sim2real: bool,
    num_laps: int,
    map_name: str | None,
) -> dict[int, list[LayoutResult]]:
    """One batched policy.run per seed."""
    from evaluation import policy

    per_seed: dict[int, list[LayoutResult]] = {}
    for seed in seeds:
        cfg, agent = _cached_agent(run, seed, checkpoint)
        results = policy.run(
            agent, on_map(run, cfg, map_name), layouts, sim2real=sim2real, num_laps=num_laps
        )
        for result, index in zip(results, indices, strict=True):
            result.index = index
            _progress(seed, result)
        per_seed[seed] = list(results)
    return per_seed


def _run_filter(
    run: str,
    seeds: Sequence[int],
    checkpoint: str,
    layouts: Sequence[Layout],
    indices: Sequence[int],
    *,
    sf: str,
    sim2real: bool,
    obstacles: str,
    num_laps: int,
    jobs: int,
    log: bool,
    render: bool,
    slowdown: float,
    out_dir: Path,
    map_name: str | None,
    driver: str = "policy",
    speed_scale: float = 1.0,
) -> tuple[dict[int, list[LayoutResult]], list[Path]]:
    """Tick-loop one layout at a time, optionally fanned over a spawn pool."""
    items: list[dict[str, Any]] = []
    logged: list[Path] = []
    for seed in seeds:
        for layout, index in zip(layouts, indices, strict=True):
            log_path = out_dir / f"filter_log_seed{seed}_k{index}.csv" if log else None
            if log_path is not None:
                logged.append(log_path)
            items.append(
                {
                    "run": run,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "layout": layout,
                    "index": index,
                    "sf": sf,
                    "sim2real": sim2real,
                    "obstacles": obstacles,
                    "num_laps": num_laps,
                    "log_path": str(log_path) if log_path is not None else None,
                    "render": render,
                    "slowdown": slowdown,
                    "map": map_name,
                    "driver": driver,
                    "speed_scale": speed_scale,
                }
            )
    per_seed: dict[int, list[LayoutResult]] = {seed: [] for seed in seeds}
    if jobs <= 1:
        for item in items:
            seed, result = _filter_job(item)
            _progress(seed, result)
            per_seed[seed].append(result)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=ctx, initializer=_init_worker
        ) as pool:
            futures = [pool.submit(_filter_job, item) for item in items]
            for fut in as_completed(futures):
                seed, result = fut.result()
                _progress(seed, result)
                per_seed[seed].append(result)
    for seed in per_seed:
        per_seed[seed].sort(key=lambda result: result.index)
    return per_seed, logged


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ./eval <run> [--sf on|shadow|off] [--seeds] [--layout] [--render]."""
    from evaluation import report

    args = _parse_args(argv)
    cfg = load(resolve(args.run))
    run_dir = REPO / "outputs" / "trained_agents" / args.run
    suffix = _checkpoint_suffix(args.checkpoint)
    seeds = args.seeds if args.seeds is not None else _discover_seeds(run_dir, suffix)
    if not seeds:
        raise SystemExit(
            f"no _actor_seed_*{suffix} files in {run_dir} for checkpoint {args.checkpoint!r}"
        )
    reactive = args.driver != "policy"
    if reactive:
        # The weights are not used; one seed labels the run.
        seeds = seeds[:1]
    trained_map = cfg.track.map
    # Cross-map only when --map names another map; the run's own map is a no-op.
    map_name = args.map if args.map not in (None, trained_map) else None
    if map_name is not None:
        cfg = on_map(args.run, cfg, map_name)
        eval_set = args.eval_set or "empty"
    else:
        eval_set = args.eval_set or cfg.training.eval_set or "empty"
    if eval_set != "empty":
        got = set_map(eval_set)
        if got != cfg.track.map:
            raise SystemExit(
                f"eval set {eval_set!r} is map {got!r}, evaluating on {cfg.track.map!r}"
                + ("" if map_name else f" (run {args.run!r}'s map; --map overrides)")
            )
    layouts, indices = _layouts(eval_set, args.layout)
    if args.sf != "off" or args.render or reactive:
        try:
            import evaluation.filter  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name not in _FILTER_MODULES:
                raise
            mode = f"--sf {args.sf}" if args.sf != "off" else "--render"
            raise SystemExit(
                f"{mode} needs the safety filter ({exc.name} not found: run ./rebuild); "
                "use --sf off without --render"
            ) from exc
    num_laps = int(args.laps if args.laps is not None else cfg.training.num_laps)
    sim2real = (
        bool(args.sim2real)
        if args.sim2real is not None
        else bool(cfg.training.sim2real)
    )
    n_tasks = len(seeds) * len(layouts)
    if args.render or (args.sf == "off" and not reactive):
        jobs = 1
    else:
        requested = args.jobs if args.jobs is not None else _default_jobs()
        jobs = min(requested, max(1, n_tasks))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    on_suffix = f"_on-{map_name}" if map_name else ""
    who = f"_{args.driver}-x{args.speed_scale:g}" if reactive else ""
    out_dir = OUT_DIR / args.run / f"{stamp}_sf-{args.sf}{who}_{eval_set}{on_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"eval: run={args.run} checkpoint={args.checkpoint} "
        f"seeds={','.join(str(s) for s in seeds)} sf={args.sf} eval_set={eval_set} "
        f"map={cfg.track.map} trained_map={trained_map} "
        f"layouts={len(layouts)} laps={num_laps} sim2real={sim2real} "
        f"obstacles={args.obstacles} driver={args.driver} speed_scale={args.speed_scale:g} "
        f"jobs={jobs} out={out_dir}",
        flush=True,
    )
    logged: list[Path] = []
    if args.sf == "off" and not args.render and not reactive:
        per_seed = _run_policy(
            args.run, seeds, args.checkpoint, layouts, indices, sim2real, num_laps, map_name
        )
    else:
        per_seed, logged = _run_filter(
            args.run,
            seeds,
            args.checkpoint,
            layouts,
            indices,
            sf=args.sf,
            sim2real=sim2real,
            obstacles=args.obstacles,
            num_laps=num_laps,
            jobs=jobs,
            log=args.log,
            render=args.render,
            slowdown=float(args.slowdown),
            out_dir=out_dir,
            map_name=map_name,
            driver=args.driver,
            speed_scale=float(args.speed_scale),
        )
    eval_args = {
        "run": args.run,
        "checkpoint": args.checkpoint,
        "seeds": list(seeds),
        "sf": args.sf,
        "eval_set": eval_set,
        "map": cfg.track.map,
        "trained_map": trained_map,
        "layout": args.layout,
        "laps": num_laps,
        "sim2real": sim2real,
        "obstacles": args.obstacles,
        "driver": args.driver,
        "speed_scale": float(args.speed_scale),
        "jobs": jobs,
        "log": bool(args.log),
        "render": bool(args.render),
        "slowdown": float(args.slowdown),
    }
    summary_path = report.write(out_dir, cfg, eval_args, per_seed)
    report.print_summary(per_seed, cfg)
    for path in logged:
        if path.is_file():
            report.plot(path, path.with_suffix(".png"))
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
