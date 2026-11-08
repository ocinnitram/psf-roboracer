"""Where a logged 500-layout run crashed, and into what: writes crashes.json next to its layouts.tsv.

    uv run python -m evaluation.tools.reactive_crashes outputs/eval/<run>/<stamp>_..._obs-set-dev-500

Needs the run's tick logs (--log). For every crashed layout, from its last logged pose: the arc length, lap
and speed, the smallest distance from the 0.50 x 0.29 m footprint to a box and to the walls (the raceline
offset by w_l and w_r), and `hit`, whichever of the two is nearer. The pose is
the measured one, so the distances carry its noise (a few cm); only the comparison is used.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.common import REPO

EGO_L, EGO_W = 0.50, 0.29


def footprint(cx: float, cy: float, yaw: float) -> np.ndarray:
    """Corners, side midpoints and end midpoints of the footprint."""
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array(
        [[a * EGO_L / 2, b * EGO_W / 2] for a in (-1, 0, 1) for b in (-1, 1)]
        + [[EGO_L / 2, 0.0], [-EGO_L / 2, 0.0]]
    )
    return loc @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def box_distance(points: np.ndarray, ob: dict) -> float:
    """Smallest distance from the points to the box (zero inside)."""
    c, s = np.cos(ob["yaw"]), np.sin(ob["yaw"])
    q = (points - [ob["x"], ob["y"]]) @ np.array([[c, -s], [s, c]])
    dx = np.maximum(np.abs(q[:, 0]) - ob["half_length"], 0.0)
    dy = np.maximum(np.abs(q[:, 1]) - ob["half_width"], 0.0)
    return float(np.hypot(dx, dy).min())


def main(run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    eval_set, track_map = summary["eval"]["eval_set"], summary["eval"]["map"]
    rl = np.genfromtxt(REPO / "sf" / "tracks" / track_map / "raceline.csv", delimiter=",", names=True)
    walls = np.vstack(
        [
            np.c_[rl["x"] + sign * rl[w] * rl["n_x"], rl["y"] + sign * rl[w] * rl["n_y"]]
            for sign, w in ((1.0, "w_l"), (-1.0, "w_r"))
        ]
    )
    out = {}
    for row in csv.DictReader((run_dir / "layouts.tsv").open(), delimiter="\t"):
        if row["outcome"] != "crashed":
            continue
        k, seed = int(row["layout"]), int(row["seed"])
        with (run_dir / f"filter_log_seed{seed}_k{k}.csv").open() as f:
            last = list(csv.DictReader(f))[-1]
        p = footprint(float(last["x"]), float(last["y"]), float(last["yaw"]))
        layout = json.loads((REPO / "evaluation" / "layouts" / eval_set / f"k{k}.json").read_text())
        obs = layout["obstacles"]
        d_box = min(box_distance(p, ob) for ob in obs)
        d_wall = float(min(np.hypot(*(walls - q).T).min() for q in p))
        out[k] = {
            "s": float(last["s"]),
            "lap": int(last["lap"]),
            "v": float(last["v"]),
            "box_m": d_box,
            "wall_m": d_wall,
            "hit": "box" if d_box < d_wall else "wall",
            "box_s": [ob["s"] for ob in obs],
        }
    (run_dir / "crashes.json").write_text(json.dumps(out, indent=1))
    print(
        f"{len(out)} crashes: {dict(Counter(v['hit'] for v in out.values()))}, "
        f"{sum(v['lap'] == 0 for v in out.values())} on the first lap"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
