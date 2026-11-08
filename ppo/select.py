"""Rank a wandb sweep by converged score and write the next-stage configs.

A trial's score is the mean of its last K in-loop evaluations, not the best
one and not the last one. With `--seeded`, every trial is grouped with the
seed runs `./train --wandb` logged under its name and ranked by the mean over
seeds, worst seed as the tie-break.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

from ppo.config import SEEDS, dump, load, parse_set, resolve
from ppo.sweep import TRIAL_SEED, overrides_from_wandb

REPO = Path(__file__).resolve().parents[1]
PROJECT = "ppo-lab"
HISTORY_KEYS = (
    "score",
    "Eval/SuccessRate",
    "Eval/Pace",
    "Eval/SteerRate",
    "Eval/SpeedCmdRate",
)
FIXED_COLUMNS = (
    "name",
    "seeds",
    "evals",
    "score",
    "score_min",
    "success_rate",
    "pace",
    "steer_rate",
    "speed_cmd_rate",
)
DEFAULT_TOP = 5
DEFAULT_LAST = 10
# Score gap below which candidates are ranked by steer rate instead (a 5 % pace difference).
TIE_BAND = 0.005
# One eval per 100k steps; 1000 samples covers a 100M-step run in a single request.
HISTORY_SAMPLES = 1000
FETCH_WORKERS = 8
DEFAULT_SEEDS = [seed for seed in SEEDS if seed != TRIAL_SEED]


@dataclass(frozen=True)
class Candidate:
    """One sweep trial with the evaluation history of every seed run under its name."""

    name: str
    config: Mapping[str, Any]
    histories: tuple[tuple[Mapping[str, Any], ...], ...]


@dataclass(frozen=True)
class SeedResult:
    """Converged metrics of one seed: means over its last K evaluations."""

    score: float
    success_rate: float
    pace: float
    steer_rate: float | None
    speed_cmd_rate: float | None
    evals: int


def tsv_columns(parameter_keys: Sequence[str]) -> list[str]:
    """Return TSV column names: identity, metrics, then swept parameters."""
    return [*FIXED_COLUMNS, *parameter_keys]


def parameter_keys_from_sweep_config(config: Mapping[str, Any]) -> list[str]:
    """List the swept parameter names from a wandb sweep config."""
    params = config.get("parameters")
    if not isinstance(params, Mapping):
        return []
    return [str(key) for key in params.keys()]


def config_value(config: Mapping[str, Any], dotted: str) -> Any:
    """Read a dotted key from a run config, nested or flat."""
    if dotted in config:
        return config[dotted]
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _finite(value: Any) -> float | None:
    """Return the value as a finite float, else None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the finite values, None when there are none."""
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def converged(history: Iterable[Mapping[str, Any]], last: int) -> SeedResult | None:
    """Average the last K evaluation rows of one run; None when it never evaluated."""
    rows = [row for row in history if _finite(row.get("score")) is not None]
    tail = rows[-last:]
    if not tail:
        return None
    score = _mean(_finite(row.get("score")) for row in tail)
    success = _mean(_finite(row.get("Eval/SuccessRate")) for row in tail)
    pace = _mean(_finite(row.get("Eval/Pace")) for row in tail)
    assert score is not None
    return SeedResult(
        score=score,
        success_rate=success if success is not None else 0.0,
        pace=pace if pace is not None else 0.0,
        steer_rate=_mean(_finite(row.get("Eval/SteerRate")) for row in tail),
        speed_cmd_rate=_mean(_finite(row.get("Eval/SpeedCmdRate")) for row in tail),
        evals=len(tail),
    )


def rank(
    candidates: Sequence[Candidate],
    parameter_keys: Sequence[str],
    last: int,
) -> list[dict[str, Any]]:
    """Rank candidates by mean converged score over seeds, then worst seed."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        results = [converged(history, last) for history in candidate.histories]
        seeds = [result for result in results if result is not None]
        if not seeds:
            continue
        score = _mean(seed.score for seed in seeds)
        assert score is not None
        row: dict[str, Any] = {
            "name": candidate.name,
            "seeds": len(seeds),
            "evals": min(seed.evals for seed in seeds),
            "score": score,
            "score_min": min(seed.score for seed in seeds),
            "success_rate": _mean(seed.success_rate for seed in seeds),
            "pace": _mean(seed.pace for seed in seeds),
            "steer_rate": _mean(seed.steer_rate for seed in seeds),
            "speed_cmd_rate": _mean(seed.speed_cmd_rate for seed in seeds),
        }
        for key in parameter_keys:
            row[key] = config_value(candidate.config, key)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["score_min"]),
            -int(row["seeds"]),
            -float(row["success_rate"]),
            -float(row["pace"]),
            str(row["name"]),
        )
    )
    if not rows:
        return rows
    # Scores within TIE_BAND of the best are one candidate as far as the objective
    # can tell; among those the smoother policy (lowest steer rate) wins.
    best = float(rows[0]["score"])
    tied = [row for row in rows if float(row["score"]) >= best - TIE_BAND]
    tied.sort(key=lambda row: (round(float(row["steer_rate"] or 0.0), 4), rows.index(row)))
    return tied + rows[len(tied) :]


def write_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    parameter_keys: Sequence[str],
) -> None:
    """Write ranked rows as a tab-separated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tsv_columns(parameter_keys)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    "" if row.get(column) is None else row.get(column)
                    for column in columns
                ]
            )


def format_table(
    rows: Sequence[Mapping[str, Any]],
    parameter_keys: Sequence[str],
    top: int,
) -> str:
    """Format the top ranked rows as an aligned table."""
    columns = tsv_columns(parameter_keys)
    shown = list(rows[:top])
    if not shown:
        return "(no scored runs)"
    cells: list[list[str]] = [list(columns)]
    for row in shown:
        line: list[str] = []
        for column in columns:
            value = row.get(column)
            if value is None:
                line.append("")
            elif isinstance(value, float):
                line.append(f"{value:.4f}")
            else:
                line.append(str(value))
        cells.append(line)
    widths = [max(len(row[index]) for row in cells) for index in range(len(columns))]
    lines = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(columns)))
        for row in cells
    ]
    return "\n".join(lines)


def phase_of(run_name: str) -> str:
    """Recover the phase from a trial name `<phase>-<id>`; phases carry no dash."""
    return run_name.split("-", 1)[0]


def candidate_config(
    name: str,
    run_config: Mapping[str, Any],
    base: str | None,
    fixed: Mapping[str, Any] | None = None,
) -> str:
    """Rebuild a trial's effective config as yaml: base, the sweep's fixed --set values, its wandb overrides."""
    overrides = dict(fixed or {})
    overrides.update(overrides_from_wandb(run_config))
    overrides["training.run_name"] = name
    return dump(load(resolve(base or phase_of(name)), overrides))


def seed_tag(sweep_id: str) -> str:
    """Wandb tag that marks the seed runs of one sweep's candidates."""
    return f"seeds-{short_id(sweep_id)}"


def candidates_dir(sweep_id: str) -> Path:
    """Directory that holds the next-stage configs of one sweep."""
    return REPO / "outputs" / "sweeps" / short_id(sweep_id)


def write_candidates(
    sweep_id: str,
    rows: Sequence[Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
    base: str | None,
    fixed: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write one config yaml per ranked row and return the paths in rank order."""
    directory = candidates_dir(sweep_id)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in rows:
        name = str(row["name"])
        path = directory / f"{name}.yaml"
        path.write_text(candidate_config(name, configs[name], base, fixed))
        paths.append(path)
    return paths


def resolve_sweep_path(sweep_id: str, api: Any) -> str:
    """Turn a bare id, path, or wandb URL into entity/project/id."""
    text = sweep_id.strip().rstrip("/")
    marker = "wandb.ai/"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = text.replace("/sweeps/", "/")
    parts = [part for part in text.split("/") if part]
    if not parts:
        return sweep_id
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    entity = getattr(api, "default_entity", None)
    if len(parts) == 2:
        if entity:
            return f"{entity}/{parts[0]}/{parts[1]}"
        return f"{parts[0]}/{parts[1]}"
    if entity:
        return f"{entity}/{PROJECT}/{parts[0]}"
    return parts[0]


def short_id(sweep_id: str) -> str:
    """Return the bare sweep id from an id, path, or URL."""
    return sweep_id.strip().rstrip("/").split("/")[-1]


def tsv_path_for(sweep_id: str) -> Path:
    """Return the TSV path for a sweep id, using the short id as the file name."""
    return REPO / "outputs" / "sweeps" / f"{short_id(sweep_id)}.tsv"


def _history(run: Any) -> tuple[Mapping[str, Any], ...]:
    """Fetch the evaluation rows of one wandb run in one request."""
    rows = run.history(keys=list(HISTORY_KEYS), samples=HISTORY_SAMPLES, pandas=False)
    rows.sort(key=lambda row: row.get("_step", 0))
    return tuple(rows)


def _histories(runs: Sequence[Any]) -> list[tuple[Mapping[str, Any], ...]]:
    """Fetch run histories concurrently, in input order."""
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        return list(pool.map(_history, runs))


def collect(api: Any, sweep: Any, project_path: str, seeded: bool) -> list[Candidate]:
    """Build one candidate per sweep trial.

    With seeded, only the trials that have seed runs grouped under their name
    are candidates, each with those runs attached; a trial that never made the
    seed stage is not compared against those that did.
    """
    trials = list(sweep.runs)
    names = [str(run.name) for run in trials]
    grouped: dict[str, list[Any]] = {name: [] for name in names}
    if seeded:
        for seed_run in api.runs(project_path, filters={"group": {"$in": names}}):
            grouped[str(seed_run.group)].append(seed_run)
        trials = [trial for trial in trials if grouped[str(trial.name)]]
    runs = [run for trial in trials for run in (trial, *grouped[str(trial.name)])]
    histories = iter(_histories(runs))
    candidates: list[Candidate] = []
    for trial in trials:
        count = 1 + len(grouped[str(trial.name)])
        config = getattr(trial, "config", {}) or {}
        candidates.append(
            Candidate(
                str(trial.name),
                dict(config),
                tuple(next(histories) for _ in range(count)),
            )
        )
    return candidates


def _parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated seed list."""
    seeds = [int(part) for part in spec.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds needs at least one seed")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the select command line."""
    parser = argparse.ArgumentParser(
        prog="select",
        description="Rank a wandb sweep by converged score and write the next-stage configs",
    )
    parser.add_argument("sweep_id", help="Wandb sweep id, entity/project/id, or URL")
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP, help="Candidates to keep"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST,
        help="Evaluations averaged per run (from the end)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="Seeds the printed train commands use (trial seed excluded)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Config the trials were built on (default: the phase in the trial name)",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="Fixed override the sweep was launched with, e.g. track.map=lab-20260906",
    )
    parser.add_argument(
        "--seeded",
        action="store_true",
        help="Rank by mean over the seed runs grouped under each trial name",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.last < 1:
        parser.error("--last must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Fetch a sweep, rank it, write the TSV and next-stage configs, print the winner."""
    args = parse_args(argv)
    api = wandb.Api()
    path = resolve_sweep_path(args.sweep_id, api)
    sweep = api.sweep(path)
    project_path = path.rsplit("/", 1)[0] if "/" in path else PROJECT
    parameter_keys = parameter_keys_from_sweep_config(sweep.config)
    candidates = collect(api, sweep, project_path, args.seeded)
    rows = rank(candidates, parameter_keys, args.last)
    tsv_path = tsv_path_for(sweep.id)
    write_tsv(tsv_path, rows, parameter_keys)
    seeds = ",".join(str(seed) for seed in args.seeds)
    print(
        f"select  sweep_id={sweep.id}  last={args.last}  top={args.top}  "
        f"seeded={'on' if args.seeded else 'off'}  seeds={seeds}  "
        f"base={args.base or 'phase'}  set={','.join(args.sets) or 'none'}  "
        f"tsv={tsv_path.relative_to(REPO)}"
    )
    print(format_table(rows, parameter_keys, args.top))
    if not rows:
        raise SystemExit("no scored runs in this sweep")
    if args.seeded:
        print(rows[0]["name"])
        return
    configs = {candidate.name: candidate.config for candidate in candidates}
    paths = write_candidates(sweep.id, rows[: args.top], configs, args.base, parse_set(args.sets))
    print("next stage:")
    for row, config_path in zip(rows, paths):
        warmstart = str(configs[row["name"]].get("warmstart") or "")
        print(
            f"  ./train {config_path.relative_to(REPO)} --seeds {seeds} --wandb "
            f"--tag {seed_tag(sweep.id)}"
            + (f" --warmstart {warmstart}" if warmstart else "")
        )
    print(f"then: ./select {sweep.id} --seeded --last {args.last}")


if __name__ == "__main__":
    main(sys.argv[1:])
