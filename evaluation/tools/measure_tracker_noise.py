"""Fit detector (or tracker) centre noise against range from filter logs.

Reads ``filter_log_seed<S>_k<K>.csv`` files or every such file in an eval
output dir, and compares centres to layout crate poses, seen from the LiDAR.
The detector's fit is first moved to the crate's true depth the way the
tracker moves it (``corrected_centre``). Across-range scatter
is fit to ``sigma0^2 + (sigma_heading * range)^2``; that fit and the along
residual are the tracker constants in ``sf/include/obstacle_tracker.h``.

    uv run --frozen python evaluation/tools/measure_tracker_noise.py DIR
    uv run --frozen python evaluation/tools/measure_tracker_noise.py LOG.csv --eval-set obs-set-dev-500 --tracker
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from evaluation.common import REPO  # noqa: E402

LOG_NAME = re.compile(r"filter_log_seed\d+_k(\d+)\.csv$")
SIGMA_ALONG_M, SIGMA_ACROSS_M, SIGMA_HEADING_RAD, BIAS_ALONG_M = 0.042, 0.042, 0.012, 0.030


def _finite(value: object) -> bool:
    """True when value parses as a finite float."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _lidar_origin(row: dict[str, str], lidar_dist: float) -> np.ndarray:
    """LiDAR mounting position from the logged CoG pose (sf/stack.py lidar_pose)."""
    yaw = float(row["yaw"])
    return np.array([float(row["x"]) + lidar_dist * np.cos(yaw), float(row["y"]) + lidar_dist * np.sin(yaw)])


def corrected_centre(row: dict[str, str], crate: dict, origin: np.ndarray) -> np.ndarray:
    """Fit centre moved to the crate's true depth, as obstacle_tracker.cpp position_measurement does.

    The box axes are the crate's (u along its yaw, v across it); the depth axis is
    the one closer to the line of sight from the LiDAR. The fitted side lying along
    that axis is the measured depth (``raw_length`` runs along ``raw_yaw``), and the
    centre moves by half the difference to the true depth, away from the LiDAR, so
    the measured face stays where it was seen. The tracker's own line-of-sight bias
    correction is not applied: that bias is what is being measured.
    """
    cu, su = np.cos(crate["yaw"]), np.sin(crate["yaw"])
    cx, cy = float(row["raw_x"]), float(row["raw_y"])
    rx, ry = cx - origin[0], cy - origin[1]
    au, av = rx * cu + ry * su, -rx * su + ry * cu
    along_u = abs(au) >= abs(av)
    sign = (1.0 if au >= 0 else -1.0) if along_u else (1.0 if av >= 0 else -1.0)
    ax, ay = (cu, su) if along_u else (-su, cu)
    length_along_u = abs(np.cos(float(row["raw_yaw"]) - crate["yaw"])) >= np.sqrt(0.5)
    length, width = float(row["raw_length"]), float(row["raw_width"])
    depth_det = (length if length_along_u else width) if along_u else (width if length_along_u else length)
    depth_true = 2 * crate["half_length"] if along_u else 2 * crate["half_width"]
    corr = 0.5 * (depth_true - depth_det) * sign
    return np.array([cx + corr * ax, cy + corr * ay])


def _reported(row: dict[str, str], tracker: bool) -> np.ndarray | None:
    """Uncorrected centre for one tick, or None when that channel is empty."""
    prefix, count = ("obs", "n_tracked") if tracker else ("raw", "n_raw")
    if not _finite(row.get(count, "")) or float(row[count]) < 1:
        return None
    need = [f"{prefix}_x", f"{prefix}_y"] + ([] if tracker else ["raw_yaw", "raw_length", "raw_width"])
    if not all(_finite(row.get(k, "")) for k in need):
        return None
    return np.array([float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])])


def plant_crates(manifest: dict) -> list[dict]:
    """Crate world poses exactly as the plant places them (roboracer.h place_obstacle_in_world).

    The plant interpolates the raceline at the crate's station and offsets it by
    ``n`` along the left normal. The manifest's own ``x, y`` snap to the nearest
    raceline row instead, which puts them about 1 cm further along the track, so
    they are not used as the truth.
    """
    data = np.loadtxt(REPO / "sf" / "tracks" / manifest["map"] / "raceline.csv", delimiter=",", skiprows=1)
    xs, ys, ss, yaws = data[:, 0], data[:, 1], data[:, 2], data[:, 4]
    lap = ss[-1] + np.hypot(xs[0] - xs[-1], ys[0] - ys[-1])
    crates = []
    for c in manifest["obstacles"]:
        i = int(np.searchsorted(ss, c["s"], side="right") - 1)
        j = (i + 1) % len(ss)
        f = (c["s"] - ss[i]) / ((lap if j == 0 else ss[j]) - ss[i])
        yaw = yaws[i] + f * ((yaws[j] - yaws[i] + np.pi) % (2 * np.pi) - np.pi)
        x = xs[i] + f * (xs[j] - xs[i]) - c["n"] * np.sin(yaw)
        y = ys[i] + f * (ys[j] - ys[i]) + c["n"] * np.cos(yaw)
        crates.append({"x": x, "y": y, "yaw": yaw, "half_length": c["half_length"], "half_width": c["half_width"]})
    return crates


def errors(files: list[Path], eval_set: str, tracker: bool) -> np.ndarray:
    """Line-of-sight residuals (range, along, across) from the LiDAR against the nearest crate."""
    lidar_dist = float(yaml.safe_load((REPO / "vehicle" / "params.yaml").read_text())["lidar_dist_m"])
    rows: list[list[float]] = []
    layouts = REPO / "evaluation" / "layouts" / eval_set
    for path in files:
        match = LOG_NAME.search(path.name)
        if match is None:
            continue
        manifest = layouts / f"k{match.group(1)}.json"
        if not manifest.is_file():
            continue
        crates = plant_crates(json.loads(manifest.read_text()))
        truth = np.array([[c["x"], c["y"]] for c in crates], dtype=float)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if len(truth) == 0 or not all(_finite(row.get(k)) for k in ("x", "y", "yaw")):
                    continue
                reported = _reported(row, tracker)
                if reported is None:
                    continue
                origin = _lidar_origin(row, lidar_dist)
                i = int(np.argmin(np.linalg.norm(truth - reported, axis=1)))
                z = reported if tracker else corrected_centre(row, crates[i], origin)
                if np.linalg.norm(truth[i] - z) > 0.8:
                    continue
                rx, ry = z - origin
                rng = float(np.hypot(rx, ry))
                if rng < 1e-9:
                    continue
                u = np.array([rx, ry]) / rng
                e = z - truth[i]
                rows.append([rng, float(e @ u), float(e @ np.array([-u[1], u[0]]))])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def report(data: np.ndarray) -> None:
    """Print the per-metre table, the sigma fit, and the header constants."""
    rng, along, across = data[:, 0], data[:, 1], data[:, 2]
    bins = np.minimum(np.floor(rng).astype(int), 8)
    print(f"{'bin':>3} {'n':>6} {'along_mean':>11} {'along_sd':>9} "
          f"{'across_mean':>12} {'across_sd':>10} {'across_p99':>11} {'norm_p90':>9}")
    r2, var = [], []
    for b in range(9):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        al, ac = along[mask], across[mask]
        ddof = 1 if n > 1 else 0
        print(f"{b:3d} {n:6d} {al.mean():11.3f} {al.std(ddof=ddof):9.3f} "
              f"{ac.mean():12.3f} {ac.std(ddof=ddof):10.3f} "
              f"{np.percentile(np.abs(ac), 99):11.3f} {np.percentile(np.hypot(al, ac), 90):9.3f}")
        if n >= 2:
            r2.append(float(rng[mask].mean() ** 2))
            var.append(float(np.var(ac, ddof=1)))
    if len(r2) >= 2:
        coef, *_ = np.linalg.lstsq(np.vstack([np.ones(len(r2)), r2]).T, np.asarray(var), rcond=None)
        sigma0, heading = np.sqrt(max(coef[0], 0.0)), np.sqrt(max(coef[1], 0.0))
    else:
        sigma0, heading = float("nan"), float("nan")
    print(f"across: sigma0 {sigma0:.3f} m, sigma_heading {heading:.4f} rad; "
          f"along: sd {along.std(ddof=1):.3f} m, bias {along.mean():+.3f} m  ({len(data)} ticks)")
    print(f"constants to set (obstacle_tracker.h): kSigmaAlongM={SIGMA_ALONG_M} "
          f"kSigmaAcrossM={SIGMA_ACROSS_M} kSigmaHeadingRad={SIGMA_HEADING_RAD} kBiasAlongM={BIAS_ALONG_M}")


def main(argv: list[str] | None = None) -> None:
    """Load logs, print the per-range table and the tracker-noise fit."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="filter_log csv(s) or an eval output dir")
    ap.add_argument("--eval-set", default=None, help="layout set; default from summary.json when a dir is given")
    ap.add_argument("--tracker", action="store_true", help="score obs_* columns instead of corrected raw_*")
    args = ap.parse_args(argv)
    files, dirs = [], []
    for path in args.paths:
        if path.is_dir():
            dirs.append(path)
            files.extend(sorted(path.glob("filter_log_*.csv")))
        else:
            files.append(path)
    name = args.eval_set
    if not name:
        for directory in dirs:
            summary = directory / "summary.json"
            if summary.is_file():
                name = str(json.loads(summary.read_text())["eval"]["eval_set"])
                break
    if not name:
        raise SystemExit("pass --eval-set, or an eval output dir that contains summary.json")
    data = errors(files, name, args.tracker)
    if data.size == 0:
        raise SystemExit(f"no detections in {len(files)} log(s) for eval-set {name}")
    report(data)


if __name__ == "__main__":
    main()
