"""JSON summary, TSV, printed table, and filter-log plots."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from evaluation.common import CRASHED, LayoutResult
from ppo.config import Config, score

_NUMERIC = (
    "success_rate", "pace", "mean_lap_time", "collisions", "n_layouts",
    "steer_rate", "speed_cmd_rate", "interventions", "speed_capped", "uncertified", "score",
)
_TABLE = (
    ("seed", "seed"), ("success", "success_rate"), ("pace", "pace"),
    ("lap", "mean_lap_time"), ("coll", "collisions"), ("interv", "interventions"),
    ("cap", "speed_capped"), ("uncert", "uncertified"), ("score", "score"),
)
_OUTCOME_COLOR = {
    "Passthrough": "0.6", "Qp": "tab:orange", "Segment": "tab:orange",
    "Backup": "tab:orange", "Uncertified": "tab:red", "": "tab:blue",
}
_TSV = (
    "seed", "layout", "n_obstacles", "outcome", "laps_s",
    "interventions", "uncertified", "term_v", "steps", "speed_capped",
)
_OBS = ("obs_x", "obs_y", "obs_yaw", "obs_length", "obs_width")


def _uncertified(result: LayoutResult) -> int:
    """Count Uncertified filter ticks on one layout."""
    return int(result.outcomes.get("Uncertified", 0))


def _sum_counts(results: Sequence[LayoutResult], attr: str) -> dict[str, int]:
    """Sum per-layout string-count dicts."""
    total: Counter[str] = Counter()
    for result in results:
        total.update(getattr(result, attr))
    return dict(total)


def summarise(results: Sequence[LayoutResult], cfg: Config) -> dict:
    """Aggregate one seed's layouts into the printed and JSON metrics."""
    n = len(results)
    completed = [r for r in results if r.success]
    means = [float(np.mean(r.lap_times)) for r in completed if r.lap_times]
    mean_lap = float(np.mean(means)) if means else float("nan")
    pace = float(cfg.track.opt_time) / mean_lap if means else float("nan")
    success_rate = (len(completed) / n) if n else 0.0
    total_steps = int(sum(r.steps for r in results))
    if total_steps > 0:
        steer = float(sum(r.steer_rate * r.steps for r in results) / total_steps)
        speed = float(sum(r.speed_cmd_rate * r.steps for r in results) / total_steps)
    else:
        steer = speed = 0.0
    return {
        "success_rate": success_rate, "pace": pace, "mean_lap_time": mean_lap,
        "collisions": float(sum(1 for r in results if r.outcome == CRASHED)),
        "n_layouts": float(n), "steer_rate": steer, "speed_cmd_rate": speed,
        "interventions": int(sum(r.interventions for r in results)),
        "speed_capped": int(sum(r.speed_capped for r in results)),
        "uncertified": int(sum(_uncertified(r) for r in results)),
        "outcomes": _sum_counts(results, "outcomes"),
        "selections": _sum_counts(results, "selections"),
        "score": score(success_rate, pace),
    }


def _mean_block(summaries: Sequence[dict]) -> dict[str, Any]:
    """Mean of numeric summary keys over seeds, plus min_success_rate."""
    out: dict[str, Any] = {}
    for key in _NUMERIC:
        values = np.asarray([row[key] for row in summaries], dtype=np.float64)
        out[key] = float(np.nanmean(values)) if len(values) else 0.0
    rates = [float(row["success_rate"]) for row in summaries]
    out["min_success_rate"] = min(rates) if rates else 0.0
    return out


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None so dumps writes null."""
    if isinstance(obj, dict):
        return {key: _json_safe(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(val) for val in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write(
    out_dir: Path, cfg: Config, args: dict, per_seed: dict[int, list[LayoutResult]]
) -> Path:
    """Write summary.json and layouts.tsv; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_block: dict[str, Any] = {}
    summaries: list[dict] = []
    tsv_rows: list[list[object]] = []
    for seed in sorted(per_seed):
        results = per_seed[seed]
        summary = summarise(results, cfg)
        summaries.append(summary)
        seed_block[str(seed)] = {**summary, "layouts": [r.to_dict() for r in results]}
        for r in results:
            tsv_rows.append([
                seed, r.index, r.n_obstacles, r.outcome,
                ",".join(f"{t:.4f}" for t in r.lap_times), r.interventions,
                _uncertified(r), r.terminal_v, r.steps, r.speed_capped,
            ])
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(_json_safe({
        "config": cfg.model_dump(), "eval": args, "seeds": seed_block,
        "mean": _mean_block(summaries),
    }), indent=2, default=str, allow_nan=False) + "\n")
    with (out_dir / "layouts.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(_TSV)
        writer.writerows(tsv_rows)
    return json_path


def print_summary(per_seed: dict[int, list[LayoutResult]], cfg: Config) -> None:
    """Print an aligned per-seed table plus a mean row."""
    rows: list[dict[str, Any]] = []
    summaries: list[dict] = []
    for seed in sorted(per_seed):
        summary = summarise(per_seed[seed], cfg)
        summaries.append(summary)
        rows.append({"seed": str(seed), **summary})
    if summaries:
        rows.append({"seed": "mean", **_mean_block(summaries)})
    keys = [key for _, key in _TABLE]
    cells: list[list[str]] = [[title for title, _ in _TABLE]]
    for row in rows:
        line: list[str] = []
        for key in keys:
            value = row.get(key)
            if value is None:
                line.append("")
            elif isinstance(value, float):
                line.append("" if not math.isfinite(value) else f"{value:.4f}")
            else:
                line.append(str(value))
        cells.append(line)
    widths = [max(len(row[i]) for row in cells) for i in range(len(keys))]
    print("\n".join(
        "  ".join(row[i].ljust(widths[i]) for i in range(len(keys))) for row in cells
    ), flush=True)


def plot(log_path: Path, out_png: Path) -> Path:
    """Draw trajectory by outcome and rl vs sf commands vs time."""
    from evaluation.log import read

    data = read(log_path)
    fig, (ax_xy, ax_u) = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    colors = [_OUTCOME_COLOR.get(str(o), "tab:blue") for o in data["outcome"]]
    ax_xy.scatter(data["x"], data["y"], c=colors, s=6, linewidths=0)
    seen: set[tuple[float, ...]] = set()
    for i in range(len(data["obs_x"])):
        box = tuple(float(data[key][i]) for key in _OBS)
        if not all(np.isfinite(v) for v in box):
            continue
        key = tuple(round(v, 4) for v in box)
        if key in seen:
            continue
        seen.add(key)
        cx, cy, yaw, length, width = box
        hl, hw, c, s = length / 2.0, width / 2.0, np.cos(yaw), np.sin(yaw)
        body = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        corners = body @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])
        ax_xy.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="k", lw=0.8))
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    t = data["sim_time"]
    ax_u.plot(t, data["rl_delta"], label="rl_delta", color="C0")
    ax_u.plot(t, data["sf_delta"], label="sf_delta", color="C1")
    ax_u.plot(t, data["rl_v"], label="rl_v", color="C0", ls="--")
    ax_u.plot(t, data["sf_v"], label="sf_v", color="C1", ls="--")
    ax_u.set_xlabel("sim time [s]")
    ax_u.legend(loc="upper right", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
