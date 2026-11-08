"""Error of the filter's unmeasured-state estimate against the plant.

Reads the ``true_*``, ``est_*`` and ``ss_*`` columns of ``filter_log_seed<S>_k<K>.csv``
files (``./eval ... --sf on --log``): plant truth, the carried-over estimate the
rollouts started from, and the steady-state completion of the same measured
state. Each log's first tick is dropped (the estimate is the steady state
there) and so are ticks below the kinematic floor. Errors are binned by the
measured yaw rate.

    uv run --frozen python evaluation/tools/observer_error.py DIR [DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from evaluation.log import HIDDEN, read  # noqa: E402

KINEMATIC_FLOOR_MPS = 0.5  # sf_model.h kKinematicFloorMps
YAW_BINS_RADPS = (0.0, 1.0, 2.0, 3.0, 4.0, np.inf)
# Reported units: beta in degrees, forces in N, rates as logged.
SCALE = {"beta": 180.0 / np.pi, "fy_f": 1.0, "fy_r": 1.0, "delta_dot": 1.0, "accel": 1.0}
UNIT = {"beta": "deg", "fy_f": "N", "fy_r": "N", "delta_dot": "rad/s", "accel": "m/s^2"}


def _logs(paths: list[Path]) -> list[Path]:
    """Expand eval output dirs to their filter logs."""
    out: list[Path] = []
    for path in paths:
        out.extend(sorted(path.glob("filter_log_seed*_k*.csv")) if path.is_dir() else [path])
    return out


def load(paths: list[Path]) -> dict[str, np.ndarray]:
    """Concatenate the usable ticks of every log."""
    keep = ["v", "r", *(f"{p}_{n}" for p in ("true", "est", "ss") for n in HIDDEN)]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keep}
    for path in _logs(paths):
        d = read(path)
        mask = np.isfinite(d["est_beta"]) & (d["v"] >= KINEMATIC_FLOOR_MPS)
        mask[:1] = False
        for k in keep:
            parts[k].append(d[k][mask])
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in parts.items()}


def _stats(err: np.ndarray) -> dict[str, float]:
    """RMS, mean (bias) and 99th-percentile magnitude."""
    if err.size == 0:
        return {"rms": np.nan, "bias": np.nan, "p99": np.nan}
    return {
        "rms": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "p99": float(np.percentile(np.abs(err), 99)),
    }


def summarise(d: dict[str, np.ndarray]) -> dict:
    """Per-state error of estimate and steady state, overall and per |r| bin."""
    yaw = np.abs(d["r"])
    bins = list(zip(YAW_BINS_RADPS[:-1], YAW_BINS_RADPS[1:]))
    out: dict = {"ticks": int(yaw.size), "bins": []}
    for lo, hi in [(0.0, np.inf), *bins]:
        m = (yaw >= lo) & (yaw < hi)
        row: dict = {"r_lo": lo, "r_hi": hi, "ticks": int(m.sum()), "states": {}}
        for name in HIDDEN:
            truth = d[f"true_{name}"][m] * SCALE[name]
            row["states"][name] = {
                "true_rms": float(np.sqrt(np.mean(truth**2))) if truth.size else np.nan,
                "est": _stats(d[f"est_{name}"][m] * SCALE[name] - truth),
                "ss": _stats(d[f"ss_{name}"][m] * SCALE[name] - truth),
            }
        out["bins"].append(row)
    return out


def _print(summary: dict) -> None:
    """Table: one block per state, rows per yaw-rate bin."""
    print(f"ticks {summary['ticks']}")
    for name in HIDDEN:
        print(f"\n{name} [{UNIT[name]}]")
        print(f"{'|r| rad/s':>10} {'ticks':>7} {'|true| rms':>10} "
              f"{'est rms':>8} {'est bias':>9} {'est p99':>8} "
              f"{'ss rms':>8} {'ss bias':>9} {'ss p99':>8}")
        for row in summary["bins"]:
            label = "all" if row["r_hi"] == np.inf and row["r_lo"] == 0.0 else (
                f"{row['r_lo']:.0f}+" if row["r_hi"] == np.inf else f"{row['r_lo']:.0f}-{row['r_hi']:.0f}")
            s = row["states"][name]
            e, q = s["est"], s["ss"]
            print(f"{label:>10} {row['ticks']:>7} {s['true_rms']:>10.3f} "
                  f"{e['rms']:>8.3f} {e['bias']:>9.3f} {e['p99']:>8.3f} "
                  f"{q['rms']:>8.3f} {q['bias']:>9.3f} {q['p99']:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="eval output dirs or filter logs")
    parser.add_argument("--json", type=Path, help="also write the summary here")
    args = parser.parse_args()
    summary = summarise(load(args.paths))
    _print(summary)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
