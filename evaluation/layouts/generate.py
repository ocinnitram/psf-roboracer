"""Build screened evaluation layout sets under evaluation/layouts/."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ppo.layouts import Layout, Obstacle, Track, save_set  # noqa: E402

_TRACKS = _REPO / "sf" / "tracks"
_LAYOUTS = _REPO / "evaluation" / "layouts"

MIN_S_DISTANCE_M = 5.0
START_EXCLUSION_M = 5.0
N_OBSTACLES = 2
SIGNATURE_DECIMALS = 3


@dataclass(frozen=True)
class ScreeningConfig:
    """Numbers that decide which (s, n) pairs are accepted as candidates."""

    s_step_m: float
    n_step_m: float
    conditioning_approach_m: float
    clearance_after_m: float
    max_abs_curvature_per_m: float | None
    min_track_width_m: float | None
    max_width_variation_m: float | None
    extra_reserve_m: float | None
    boundary_buffer_m: float
    collision_margin_m: float
    passing_margin_m: float
    ego_width_m: float
    ego_length_m: float
    obstacle_width_m: float
    obstacle_length_m: float
    visibility_approach_samples_m: tuple[float, ...]
    valid_zones_s: tuple[tuple[float, float], ...] | None = None

    @property
    def ego_circle_radius_m(self) -> float:
        """Radius of one quarter-body cover disk used in the reserve padding."""
        return float(np.hypot(self.ego_length_m / 4.0, self.ego_width_m / 2.0))

    @property
    def obstacle_padding_m(self) -> float:
        """Keep-out added to the obstacle half-width when measuring pass room."""
        return (
            self.obstacle_width_m / 2.0
            + self.ego_circle_radius_m
            + self.collision_margin_m
            + self.passing_margin_m
        )


@dataclass(frozen=True)
class PlacementCandidate:
    """One accepted (s, n) placement with the screening numbers at that station."""

    s: float
    n: float
    x: float
    y: float
    yaw: float
    left_reserve_m: float
    right_reserve_m: float
    max_abs_curvature_per_m: float
    min_track_width_m: float
    width_variation_m: float
    classification: str
    intended_side: int

    def as_obstacle(self, cfg: ScreeningConfig) -> Obstacle:
        """Turn this candidate into a layout obstacle."""
        return Obstacle(
            s=self.s,
            n=self.n,
            half_length=cfg.obstacle_length_m / 2.0,
            half_width=cfg.obstacle_width_m / 2.0,
        )


def load_screening_config(path: Path) -> tuple[dict[str, Any], ScreeningConfig]:
    """Read a screening.json document and the nested screening numbers."""
    with path.open() as handle:
        document = json.load(handle)
    raw = dict(document["screening"])
    raw["visibility_approach_samples_m"] = tuple(
        raw["visibility_approach_samples_m"]
    )
    zones = raw.get("valid_zones_s")
    raw["valid_zones_s"] = (
        tuple((float(lo), float(hi)) for lo, hi in zones) if zones else None
    )
    return document, ScreeningConfig(**raw)


def _nearest_index(s_track: np.ndarray, s: float) -> int:
    """Return the raceline index closest to station `s`."""
    return int(np.argmin(np.abs(s_track - s)))


def _window_indices(s_track: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Return sample indices whose stations sit in `[start_s, end_s]`."""
    return np.flatnonzero((s_track >= start_s) & (s_track <= end_s))


def enumerate_candidates(
    track: Track, cfg: ScreeningConfig, start_exclusion_m: float
) -> list[PlacementCandidate]:
    """Enumerate (s, n) pairs that pass width, curvature, and reserve screens."""
    s_track = track.s
    s_stop = float(s_track[-1] - cfg.clearance_after_m)
    zones = cfg.valid_zones_s or ((start_exclusion_m, s_stop),)
    spans: list[np.ndarray] = []
    for zone_start, zone_stop in zones:
        lo = max(float(zone_start), start_exclusion_m)
        hi = min(float(zone_stop), s_stop)
        if hi < lo:
            continue
        spans.append(np.arange(lo, hi + 0.5 * cfg.s_step_m, cfg.s_step_m))
    s_values = np.concatenate(spans) if spans else np.empty(0)

    candidates: list[PlacementCandidate] = []
    for s in s_values:
        window = _window_indices(
            s_track, s - cfg.conditioning_approach_m, s + cfg.clearance_after_m
        )
        if window.size < 2:
            continue
        max_curvature = float(np.max(np.abs(track.kappa[window])))
        widths = track.w_l[window] + track.w_r[window]
        min_width = float(np.min(widths))
        width_variation = float(np.max(widths) - np.min(widths))
        if (
            (
                cfg.max_abs_curvature_per_m is not None
                and max_curvature > cfg.max_abs_curvature_per_m
            )
            or (
                cfg.min_track_width_m is not None
                and min_width < cfg.min_track_width_m
            )
            or (
                cfg.max_width_variation_m is not None
                and width_variation > cfg.max_width_variation_m
            )
        ):
            continue

        centre_index = _nearest_index(s_track, float(s))
        obstacle_window = _window_indices(
            s_track,
            float(s) - 0.5 * (cfg.obstacle_length_m + cfg.ego_length_m),
            float(s) + 0.5 * (cfg.obstacle_length_m + cfg.ego_length_m),
        )
        if obstacle_window.size == 0:
            continue
        left_limit = float(
            np.min(track.w_l[obstacle_window] - cfg.boundary_buffer_m)
        )
        right_limit = float(
            np.max(-track.w_r[obstacle_window] + cfg.boundary_buffer_m)
        )
        obstacle_n_min = float(
            np.max(-track.w_r[obstacle_window] + cfg.obstacle_width_m / 2.0)
        )
        obstacle_n_max = float(
            np.min(track.w_l[obstacle_window] - cfg.obstacle_width_m / 2.0)
        )
        if obstacle_n_min > obstacle_n_max:
            continue

        n_values = np.arange(
            obstacle_n_min,
            obstacle_n_max + 0.5 * cfg.n_step_m,
            cfg.n_step_m,
        )
        centre_x = float(track.x[centre_index])
        centre_y = float(track.y[centre_index])
        nx = float(track.n_x[centre_index])
        ny = float(track.n_y[centre_index])
        yaw = float(track.yaw[centre_index])
        for n in n_values:
            n_f = float(n)
            x = centre_x + n_f * nx
            y = centre_y + n_f * ny
            left_reserve = left_limit - (n_f + cfg.obstacle_padding_m)
            right_reserve = (n_f - cfg.obstacle_padding_m) - right_limit
            if cfg.extra_reserve_m is None:
                classification = "unscreened"
                intended_side = 0
            else:
                left_ok = left_reserve >= cfg.extra_reserve_m
                right_ok = right_reserve >= cfg.extra_reserve_m
                if left_ok and right_ok:
                    classification = "two_sided"
                    intended_side = 0
                elif left_ok:
                    classification = "one_sided_left"
                    intended_side = 1
                elif right_ok:
                    classification = "one_sided_right"
                    intended_side = -1
                else:
                    continue
            candidates.append(
                PlacementCandidate(
                    s=float(s),
                    n=n_f,
                    x=x,
                    y=y,
                    yaw=yaw,
                    left_reserve_m=float(left_reserve),
                    right_reserve_m=float(right_reserve),
                    max_abs_curvature_per_m=max_curvature,
                    min_track_width_m=min_width,
                    width_variation_m=width_variation,
                    classification=classification,
                    intended_side=intended_side,
                )
            )
    return candidates


def station_groups(
    candidates: Iterable[PlacementCandidate], precision: int = 3
) -> dict[float, list[PlacementCandidate]]:
    """Group candidates by station rounded to `precision` decimals."""
    grouped: dict[float, list[PlacementCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(round(candidate.s, precision), []).append(candidate)
    return grouped


def layout_signature(layout: Layout) -> tuple[tuple[float, float], ...]:
    """Rounded (s, n) sequence used to keep sets disjoint."""
    return tuple(
        (round(item.s, SIGNATURE_DECIMALS), round(item.n, SIGNATURE_DECIMALS))
        for item in layout
    )


def existing_signatures(root: Path | None = None) -> set[tuple[tuple[float, float], ...]]:
    """Collect signatures from every set under evaluation/layouts/."""
    base = root if root is not None else _LAYOUTS
    forbidden: set[tuple[tuple[float, float], ...]] = set()
    for manifest_path in base.glob("*/*.json"):
        if not manifest_path.name.startswith("k"):
            continue
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        specs = manifest.get("obstacles")
        if not specs:
            continue
        forbidden.add(
            tuple(
                (
                    round(float(spec["s"]), SIGNATURE_DECIMALS),
                    round(float(spec["n"]), SIGNATURE_DECIMALS),
                )
                for spec in specs
            )
        )
    return forbidden


def _choose_candidate(
    candidates: list[PlacementCandidate],
    rng: random.Random,
    station_use: Counter[float],
) -> PlacementCandidate:
    """Pick a candidate from the least-used stations in the pool."""
    groups = station_groups(candidates)
    least_used = min(station_use[s] for s in groups)
    stations = [s for s in groups if station_use[s] <= least_used + 1]
    station = rng.choice(stations)
    return rng.choice(groups[station])


def build_uniform_layouts(
    candidates: list[PlacementCandidate],
    n_layouts: int,
    n_obstacles: int,
    min_s_distance_m: float,
    seed: int,
    forbidden: set[tuple[tuple[float, float], ...]],
) -> list[list[PlacementCandidate]]:
    """Sample stations uniformly, then n uniformly within each station."""
    groups = station_groups(candidates)
    stations = sorted(groups)
    rng = random.Random(seed)
    generated: list[list[PlacementCandidate]] = []
    for layout in range(n_layouts):
        for _ in range(10_000):
            picked: list[PlacementCandidate] = []
            for _ in range(n_obstacles):
                eligible = [
                    s
                    for s in stations
                    if all(abs(s - item.s) >= min_s_distance_m for item in picked)
                ]
                if not eligible:
                    break
                station = rng.choice(eligible)
                picked.append(rng.choice(groups[station]))
            if len(picked) != n_obstacles:
                continue
            picked.sort(key=lambda candidate: candidate.s)
            sig = tuple(
                (round(c.s, SIGNATURE_DECIMALS), round(c.n, SIGNATURE_DECIMALS))
                for c in picked
            )
            if sig in forbidden:
                continue
            forbidden.add(sig)
            generated.append(picked)
            break
        else:
            raise RuntimeError(
                f"could not construct layout {layout} with "
                f"{n_obstacles} obstacles at min_s_distance={min_s_distance_m}"
            )
    return generated


def build_class_layouts(
    candidates: list[PlacementCandidate],
    n_layouts: int,
    n_obstacles: int,
    min_s_distance_m: float,
    seed: int,
    forbidden: set[tuple[tuple[float, float], ...]],
) -> list[list[PlacementCandidate]]:
    """Build two-obstacle layouts with a 70/30 one-sided vs two-sided mix."""
    if n_obstacles != 2:
        raise ValueError("class mix is defined for two obstacles")
    total = n_layouts * n_obstacles
    one_sided_total = int(round(total * 0.7))
    if one_sided_total % 2:
        one_sided_total -= 1
    targets = (
        ["one_sided_left"] * (one_sided_total // 2)
        + ["one_sided_right"] * (one_sided_total // 2)
        + ["two_sided"] * (total - one_sided_total)
    )
    rng = random.Random(seed)
    rng.shuffle(targets)
    by_class = {
        name: [c for c in candidates if c.classification == name]
        for name in set(targets)
    }
    missing = [name for name, pool in by_class.items() if not pool]
    if missing:
        raise RuntimeError(f"screening produced no candidates for {missing}")

    station_use: Counter[float] = Counter()
    generated: list[list[PlacementCandidate]] = []
    for layout in range(n_layouts):
        requested = targets[n_obstacles * layout : n_obstacles * (layout + 1)]
        for _ in range(1_000):
            first = _choose_candidate(by_class[requested[0]], rng, station_use)
            eligible_second = [
                candidate
                for candidate in by_class[requested[1]]
                if abs(candidate.s - first.s) >= min_s_distance_m
            ]
            if not eligible_second:
                continue
            selected = [
                first,
                _choose_candidate(eligible_second, rng, station_use),
            ]
            selected.sort(key=lambda candidate: candidate.s)
            sig = tuple(
                (round(c.s, SIGNATURE_DECIMALS), round(c.n, SIGNATURE_DECIMALS))
                for c in selected
            )
            if sig in forbidden:
                continue
            forbidden.add(sig)
            generated.append(selected)
            for candidate in selected:
                station_use[round(candidate.s, SIGNATURE_DECIMALS)] += 1
            break
        else:
            raise RuntimeError(
                f"could not construct layout {layout} with classes {requested} "
                f"and min_s_distance={min_s_distance_m}"
            )
    return generated


def _rectangle_xy(
    x: float, y: float, yaw: float, half_length: float, half_width: float
) -> np.ndarray:
    """World-frame corners of a yaw-aligned rectangle."""
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return (rot @ local.T).T + np.array([x, y], dtype=float)


def write_contact_sheet(
    destination: Path,
    track: Track,
    layouts: Sequence[Layout],
) -> None:
    """Draw every layout's rectangles on the track band into one PNG."""
    n = len(layouts)
    columns = 5 if n <= 50 else 10
    rows = int(np.ceil(n / columns)) if n else 1
    cell = 2.2 if n <= 50 else 1.1
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * cell, rows * cell), squeeze=False
    )
    left_x = track.x + track.w_l * track.n_x
    left_y = track.y + track.w_l * track.n_y
    right_x = track.x - track.w_r * track.n_x
    right_y = track.y - track.w_r * track.n_y
    band_x = np.concatenate([left_x, right_x[::-1]])
    band_y = np.concatenate([left_y, right_y[::-1]])
    for index in range(rows * columns):
        axis = axes[index // columns][index % columns]
        axis.set_aspect("equal")
        axis.axis("off")
        if index >= n:
            continue
        axis.fill(band_x, band_y, color="#eeeeee", zorder=0)
        axis.plot(track.x, track.y, color="#555555", linewidth=0.4)
        layout = layouts[index]
        for item in layout:
            x, y, yaw = track.pose(item.s, item.n)
            verts = _rectangle_xy(x, y, yaw, item.half_length, item.half_width)
            axis.add_patch(
                Polygon(verts, closed=True, facecolor="#c44e52", edgecolor="#222222", linewidth=0.4)
            )
        axis.set_title(f"k{index}", fontsize=6, pad=1)
    figure.tight_layout(pad=0.2)
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def _candidates_to_layouts(
    picked: list[list[PlacementCandidate]], cfg: ScreeningConfig
) -> list[Layout]:
    """Turn rows of placement candidates into Obstacle tuples."""
    return [
        tuple(candidate.as_obstacle(cfg) for candidate in row) for row in picked
    ]


def document_has_mix(screening_path: Path) -> bool:
    """True when the map's screening config requests a class composition."""
    if not screening_path.exists():
        return False
    with screening_path.open() as handle:
        return json.load(handle).get("main_mix") is not None


def main(argv: list[str] | None = None) -> None:
    """Generate a named evaluation set and write its contact sheet."""
    parser = argparse.ArgumentParser(
        description="Generate screened evaluation obstacle layouts"
    )
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--layouts", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map", dest="map_name", default="lab-20260906")
    parser.add_argument("--n-obstacles", type=int, default=N_OBSTACLES,
                        help=f"crates per layout (default {N_OBSTACLES})")
    parser.add_argument("--min-s-distance", type=float, default=MIN_S_DISTANCE_M,
                        help=f"minimum station spacing between crates in m (default {MIN_S_DISTANCE_M})")
    args = parser.parse_args(argv)
    if args.n_obstacles != 2 and document_has_mix(_TRACKS / args.map_name / "screening.json"):
        parser.error("class-mix screening builds two-crate layouts only; use a map without main_mix")

    screening_path = _TRACKS / args.map_name / "screening.json"
    print(
        f"generate set={args.set_name} map={args.map_name} layouts={args.layouts} "
        f"seed={args.seed} min_s_distance={args.min_s_distance} "
        f"start_exclusion={START_EXCLUSION_M} n_obstacles={args.n_obstacles} "
        f"screening={screening_path}"
    )

    document, screening = load_screening_config(screening_path)
    track = Track.load(args.map_name)
    candidates = enumerate_candidates(track, screening, START_EXCLUSION_M)
    counts = Counter(c.classification for c in candidates)
    print(f"accepted {len(candidates)} candidates: {dict(counts)}")

    output_dir = _LAYOUTS / args.set_name
    existing = list(output_dir.glob("k*.json"))
    if existing:
        raise FileExistsError(
            f"{args.set_name} already has {len(existing)} manifests; "
            "evaluation sets are not overwritten"
        )

    forbidden = existing_signatures(_LAYOUTS)
    builder = (
        build_uniform_layouts
        if document.get("main_mix") is None
        else build_class_layouts
    )
    picked = builder(
        candidates,
        args.layouts,
        args.n_obstacles,
        args.min_s_distance,
        args.seed,
        forbidden,
    )
    layouts = _candidates_to_layouts(picked, screening)
    save_set(args.set_name, args.map_name, layouts, track)
    # Keep candidate world pose bits so plotting matches the screened (s, n).
    for index, row in enumerate(picked):
        path = output_dir / f"k{index}.json"
        with path.open() as handle:
            manifest = json.load(handle)
        for spec, candidate in zip(manifest["obstacles"], row):
            spec["x"] = candidate.x
            spec["y"] = candidate.y
            spec["yaw"] = candidate.yaw
            spec["s"] = candidate.s
            spec["n"] = candidate.n
        with path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

    sheet = output_dir / f"{args.set_name}.png"
    write_contact_sheet(sheet, track, layouts)
    print(f"wrote {len(layouts)} layouts and {sheet}")


if __name__ == "__main__":
    main()
