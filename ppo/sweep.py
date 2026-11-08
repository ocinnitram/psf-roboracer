"""W&B hyper-parameter sweeps for one training phase."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Mapping, Sequence

import wandb
import yaml

REPO = Path(__file__).resolve().parents[1]
SWEEPS_DIR = REPO / "ppo" / "sweeps"
PROJECT = "ppo-lab"
TRIAL_SEED = 17
DEFAULT_AGENTS = 1
DEFAULT_TRIALS = 200


@dataclass(frozen=True)
class TrialSpec:
    """Settings shared by every trial of one sweep."""

    phase: str
    base: str | None = None
    warmstart: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


def trial_run_name(phase: str, run_id: str) -> str:
    """Build the unique run name for one sweep trial."""
    return f"{phase}-{run_id}"


def overrides_from_wandb(config: Any) -> dict[str, Any]:
    """Flatten a wandb config mapping into dotted override keys."""
    raw = dict(config)
    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        head = prefix.split(".", 1)[0]
        if head.startswith("_"):
            return
        if isinstance(value, Mapping):
            for key, inner in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                walk(child, inner)
            return
        if "." in prefix:
            flat[prefix] = value

    for key, value in raw.items():
        walk(str(key), value)
    return flat


def sweep_yaml_path(phase: str) -> Path:
    """Return the sweep yaml path for a phase."""
    return SWEEPS_DIR / f"{phase}.yaml"


def load_sweep_yaml(phase: str) -> dict[str, Any]:
    """Load the wandb sweep spec for a phase."""
    path = sweep_yaml_path(phase)
    if not path.is_file():
        raise FileNotFoundError(f"sweep yaml not found: {path}")
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"sweep yaml must be a mapping: {path}")
    return loaded


def description_line(
    *,
    phase: str,
    base: str | None,
    warmstart: str | None,
    sweep_id: str | None,
    sweep_url: str | None,
    agents: int,
    trials: int,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """Format the effective sweep configuration as one line."""
    fixed = ",".join(f"{k}={v}" for k, v in (overrides or {}).items()) or "none"
    return "  ".join(
        [
            "sweep",
            f"phase={phase}",
            f"base={base or phase}",
            f"warmstart={warmstart or 'none'}",
            f"sweep_id={sweep_id or 'none'}",
            f"url={sweep_url or 'none'}",
            f"agents={agents}",
            f"trials={trials}",
            f"set={fixed}",
        ]
    )


def format_parameter_space(spec: Mapping[str, Any]) -> str:
    """Render the sweep parameter space for stdout."""
    payload = {
        "method": spec.get("method"),
        "metric": spec.get("metric"),
        "parameters": spec.get("parameters"),
    }
    if "early_terminate" in spec:
        payload["early_terminate"] = spec["early_terminate"]
    dumped = yaml.safe_dump(payload, sort_keys=False).rstrip()
    return dumped


def sweep_url_for(sweep_id: str) -> str:
    """Build a wandb UI URL from a sweep id or entity/project/id path."""
    parts = sweep_id.strip("/").split("/")
    if len(parts) >= 3:
        entity, project, sid = parts[-3], parts[-2], parts[-1]
        return f"https://wandb.ai/{entity}/{project}/sweeps/{sid}"
    return f"https://wandb.ai/{PROJECT}/sweeps/{parts[-1]}"


def trial(spec: TrialSpec) -> None:
    """Run one wandb sweep trial with a single training seed."""
    config = importlib.import_module("ppo.config")
    train = importlib.import_module("ppo.train")
    run = wandb.init(project=PROJECT)
    if run is None:
        raise RuntimeError("wandb.init returned None")
    try:
        overrides = dict(spec.overrides)
        overrides.update(overrides_from_wandb(wandb.config))
        overrides.update(train.warmstart_overrides(spec.warmstart))
        name = trial_run_name(spec.phase, str(run.id))
        overrides["training.run_name"] = name
        run.name = name
        cfg = config.load(config.resolve(spec.base or spec.phase), overrides)
        # Full effective config, so ./select can rebuild the trial without the base;
        # the warmstart so its seed stage starts from the same weights.
        run.config.update(
            {**cfg.model_dump(), "warmstart": spec.warmstart or ""}, allow_val_change=True
        )
        success_rate, pace = train.run(
            cfg, TRIAL_SEED, spec.warmstart, wandb_run=run
        )
        score = config.score(success_rate, pace)
        # Best in-loop evaluation of the trial; per-eval values are logged by train.
        run.summary["success_rate"] = float(success_rate)
        run.summary["pace"] = float(pace)
        run.summary["score"] = score
    finally:
        wandb.finish()


def agent_worker(sweep_id: str, trials: int, spec: TrialSpec) -> None:
    """Pull trials from a wandb sweep queue in one process."""
    wandb.agent(
        sweep_id,
        function=partial(trial, spec),
        count=trials,
        project=PROJECT,
    )


def physical_cores() -> int:
    """Cores the machine really has, ignoring SMT siblings."""
    logical = len(os.sched_getaffinity(0))
    smt = Path("/sys/devices/system/cpu/smt/active")
    if smt.exists() and smt.read_text().strip() == "1":
        return max(1, logical // 2)
    return logical


def threads_per_agent(agents: int) -> int:
    """Torch intra-op threads so the agents together fit in the physical cores.

    Each agent also runs the C simulator on its own thread; oversubscribing the
    machine costs more than the extra intra-op width buys.
    """
    return max(1, min(4, (physical_cores() - agents) // max(agents, 1)))


def launch_agents(sweep_id: str, agents: int, trials: int, spec: TrialSpec) -> None:
    """Start wandb agent processes with the spawn start method."""
    threads = threads_per_agent(agents)
    os.environ["TORCH_NUM_THREADS"] = str(threads)
    print(
        f"sweep  cores={physical_cores()} agents={agents} torch_threads={threads}",
        flush=True,
    )
    ctx = get_context("spawn")
    procs: list[BaseProcess] = []
    try:
        for _ in range(agents):
            proc = ctx.Process(
                target=agent_worker,
                args=(sweep_id, trials, spec),
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.join()
        raise
    failed = [proc.pid for proc in procs if proc.exitcode != 0]
    if failed:
        raise RuntimeError(f"sweep agent(s) failed: pid {failed}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the sweep command line."""
    parser = argparse.ArgumentParser(
        prog="sweep", description="Run a wandb hyper-parameter sweep"
    )
    parser.add_argument("phase", help="Training phase (empty or obstacles)")
    parser.add_argument("--base", default=None, help="Config name or trained run to load")
    parser.add_argument("--warmstart", default=None, help="Trained run whose weights to load")
    parser.add_argument(
        "--agents",
        type=int,
        default=DEFAULT_AGENTS,
        help="Concurrent wandb.agent processes",
    )
    parser.add_argument(
        "--attach",
        default=None,
        help="Existing wandb sweep id instead of creating one",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Trials each agent pulls from the sweep queue",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="Fixed dotted override for every trial, e.g. track.map=lab-20260906",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the description and parameter space, no network",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if args.agents < 1:
        parser.error("--agents must be >= 1")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Create or attach a wandb sweep and run agent processes."""
    args = parse_args(argv)
    fixed = importlib.import_module("ppo.config").parse_set(args.sets)
    spec = load_sweep_yaml(args.phase)
    print(
        description_line(
            phase=args.phase,
            base=args.base,
            warmstart=args.warmstart,
            sweep_id=args.attach,
            sweep_url=sweep_url_for(args.attach) if args.attach else None,
            agents=args.agents,
            trials=args.trials,
            overrides=fixed,
        )
    )
    if args.dry_run:
        print(format_parameter_space(spec))
        return
    sweep_id = args.attach or wandb.sweep(spec, project=PROJECT)
    if args.attach is None:
        print(
            description_line(
                phase=args.phase,
                base=args.base,
                warmstart=args.warmstart,
                sweep_id=sweep_id,
                sweep_url=sweep_url_for(sweep_id),
                agents=args.agents,
                trials=args.trials,
                overrides=fixed,
            )
        )
    launch_agents(
        sweep_id,
        args.agents,
        args.trials,
        TrialSpec(args.phase, args.base, args.warmstart, fixed),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
