"""Obstacle layouts for training and evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppo.config import ObstaclesConfig

_REPO = Path(__file__).resolve().parents[1]
_TRACKS = _REPO / "sf" / "tracks"
_LAYOUTS = _REPO / "evaluation" / "layouts"


@dataclass(frozen=True)
class Obstacle:
    """One axis-aligned track-frame rectangle."""

    s: float
    n: float
    half_length: float
    half_width: float


Layout = tuple[Obstacle, ...]


@dataclass(frozen=True)
class Zones:
    """Station intervals where obstacles may be placed."""

    intervals: tuple[tuple[float, float], ...]

    @classmethod
    def load(cls, map_name: str) -> Zones:
        """Read `valid_zones_s` from `sf/tracks/<map>/zones.json`."""
        path = _TRACKS / map_name / "zones.json"
        with path.open() as handle:
            document = json.load(handle)
        raw = document["valid_zones_s"]
        intervals = tuple((float(lo), float(hi)) for lo, hi in raw)
        return cls(intervals)


@dataclass(frozen=True)
class Track:
    """Racing-line samples used to place and plot obstacles."""

    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    w_l: np.ndarray
    w_r: np.ndarray
    n_x: np.ndarray
    n_y: np.ndarray
    kappa: np.ndarray
    length: float

    @classmethod
    def load(cls, map_name: str) -> Track:
        """Load `sf/tracks/<map>/raceline.csv`."""
        path = _TRACKS / map_name / "raceline.csv"
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
        s = data[:, 2]
        return cls(
            s=s,
            x=data[:, 0],
            y=data[:, 1],
            yaw=data[:, 4],
            w_l=data[:, 6],
            w_r=data[:, 7],
            n_x=data[:, 8],
            n_y=data[:, 9],
            kappa=data[:, 5],
            length=float(s[-1]),
        )

    def nearest_index(self, s: float) -> int:
        """Return the sample index closest to station `s`."""
        return int(np.argmin(np.abs(self.s - s)))

    def pose(self, s: float, n: float) -> tuple[float, float, float]:
        """Return world `(x, y, yaw)` of a track-frame point."""
        ndx = self.nearest_index(s)
        x = float(self.x[ndx] + n * self.n_x[ndx])
        y = float(self.y[ndx] + n * self.n_y[ndx])
        yaw = float(self.yaw[ndx])
        return x, y, yaw


def _sample_n(
    track: Track,
    s: float,
    half_width: float,
    centre_deadzone: float,
    rng: np.random.Generator,
) -> float | None:
    """Draw a lateral offset inside the local width, skipping the centre band."""
    ndx = track.nearest_index(s)
    n_lo = float(-track.w_r[ndx] + half_width)
    n_hi = float(track.w_l[ndx] - half_width)
    if n_lo > n_hi:
        return None
    dead = float(centre_deadzone)
    if dead <= 0.0:
        return float(rng.uniform(n_lo, n_hi))
    intervals: list[tuple[float, float]] = []
    left_hi = min(n_hi, -dead)
    right_lo = max(n_lo, dead)
    if n_lo < left_hi:
        intervals.append((n_lo, left_hi))
    if right_lo < n_hi:
        intervals.append((right_lo, n_hi))
    if not intervals:
        return None
    lengths = np.array([hi - lo for lo, hi in intervals], dtype=float)
    pick = int(rng.choice(len(intervals), p=lengths / lengths.sum()))
    lo, hi = intervals[pick]
    return float(rng.uniform(lo, hi))


def _sample_s_from_zones(
    intervals: Sequence[tuple[float, float]],
    num_obstacles: int,
    min_s_distance: float,
    rng: np.random.Generator,
) -> list[float]:
    """Place one station per zone first, then fill remaining slots by zone length."""
    zone_lengths = np.array([s_max - s_min for s_min, s_max in intervals], dtype=float)
    if zone_lengths.size == 0 or float(zone_lengths.sum()) <= 0.0:
        return []
    zone_weights = zone_lengths / zone_lengths.sum()
    n_zones = len(intervals)
    placed: list[float] = []
    max_attempts = 100
    zone_order = [int(i) for i in rng.permutation(n_zones)]
    for zone_idx in zone_order[: min(num_obstacles, n_zones)]:
        s_min, s_max = intervals[zone_idx]
        for _ in range(max_attempts):
            s_candidate = float(rng.uniform(s_min, s_max))
            if all(abs(s_candidate - s_prev) >= min_s_distance for s_prev in placed):
                placed.append(s_candidate)
                break
    remaining = num_obstacles - len(placed)
    for _ in range(remaining):
        for _ in range(max_attempts):
            zone_idx = int(rng.choice(n_zones, p=zone_weights))
            s_min, s_max = intervals[zone_idx]
            s_candidate = float(rng.uniform(s_min, s_max))
            if all(abs(s_candidate - s_prev) >= min_s_distance for s_prev in placed):
                placed.append(s_candidate)
                break
    placed.sort()
    return placed


def _sample_s_segments(
    track: Track,
    num_obstacles: int,
    min_s_distance: float,
    start_exclusion: float,
    obstacle_length: float,
    rng: np.random.Generator,
) -> list[float]:
    """Split the usable lap into equal segments and draw one station in each."""
    usable = track.length - start_exclusion - obstacle_length * 2.0
    if num_obstacles <= 0 or usable <= 0.0:
        return []
    segment_length = usable / num_obstacles
    placed: list[float] = []
    s_prev = 0.0
    for i in range(num_obstacles):
        s_min = max(
            s_prev + min_s_distance,
            start_exclusion + segment_length * i,
        )
        s_max = start_exclusion + segment_length * (i + 1)
        if s_min > s_max:
            continue
        s_rand = float(rng.uniform(s_min, s_max))
        placed.append(s_rand)
        s_prev = s_rand
    return placed


def sample_layout(
    track: Track,
    obstacles: ObstaclesConfig,
    zones: Zones | None,
    rng: np.random.Generator,
) -> Layout:
    """Draw a random layout on `track` from the obstacle numbers and zones."""
    half_length = float(obstacles.obstacle_length) / 2.0
    half_width = float(obstacles.obstacle_width) / 2.0
    n_want = int(obstacles.n_obstacles)
    if n_want <= 0:
        return ()
    if zones is not None and len(zones.intervals) > 0:
        clipped: list[tuple[float, float]] = []
        for s_min, s_max in zones.intervals:
            lo = max(float(s_min), float(obstacles.start_exclusion))
            hi = min(float(s_max), float(track.length))
            if hi >= lo:
                clipped.append((lo, hi))
        s_values = _sample_s_from_zones(
            clipped, n_want, float(obstacles.min_s_distance), rng
        )
    else:
        s_values = _sample_s_segments(
            track,
            n_want,
            float(obstacles.min_s_distance),
            float(obstacles.start_exclusion),
            float(obstacles.obstacle_length),
            rng,
        )
    items: list[Obstacle] = []
    for s in s_values:
        n = _sample_n(
            track, s, half_width, float(obstacles.centre_deadzone), rng
        )
        if n is None:
            continue
        items.append(
            Obstacle(
                s=float(s),
                n=float(n),
                half_length=half_length,
                half_width=half_width,
            )
        )
    items.sort(key=lambda item: item.s)
    return tuple(items)


def to_sim(layout: Layout, v_s: float = 0.0) -> list[dict[str, float]]:
    """Pack a layout for `VecEnv.set_obstacles`."""
    return [
        {
            "s": item.s,
            "n": item.n,
            "v_s": float(v_s),
            "half_length": item.half_length,
            "half_width": item.half_width,
        }
        for item in layout
    ]


def _set_dir(name: str) -> Path:
    """Return the directory that holds one named evaluation set."""
    return _LAYOUTS / name


def _index_from_stem(path: Path) -> int:
    """Parse the integer index from a `k<idx>.json` filename."""
    stem = path.stem
    if len(stem) < 2 or stem[0] != "k":
        raise ValueError(f"not a layout manifest: {path.name}")
    return int(stem[1:])


def _obstacle_from_manifest(raw: dict[str, Any]) -> Obstacle:
    """Build an Obstacle from the track-frame keys of a manifest record."""
    return Obstacle(
        s=float(raw["s"]),
        n=float(raw["n"]),
        half_length=float(raw["half_length"]),
        half_width=float(raw["half_width"]),
    )


def _obstacle_record(
    item: Obstacle, track: Track
) -> dict[str, float]:
    """Serialise an obstacle with world pose for a layout manifest."""
    x, y, yaw = track.pose(item.s, item.n)
    return {
        "s": item.s,
        "n": item.n,
        "half_length": item.half_length,
        "half_width": item.half_width,
        "x": x,
        "y": y,
        "yaw": yaw,
    }


def set_map(name: str) -> str:
    """Return the map key of `evaluation/layouts/<name>/k0.json`."""
    path = _set_dir(name) / "k0.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        return str(json.load(handle)["map"])


def load_set(name: str) -> list[Layout]:
    """Load `evaluation/layouts/<name>/k<idx>.json` in index order."""
    directory = _set_dir(name)
    paths = sorted(directory.glob("k*.json"), key=_index_from_stem)
    layouts: list[Layout] = []
    for path in paths:
        with path.open() as handle:
            manifest = json.load(handle)
        specs = manifest["obstacles"]
        layouts.append(tuple(_obstacle_from_manifest(spec) for spec in specs))
    return layouts


def save_set(
    name: str,
    map_name: str,
    layouts: Sequence[Layout],
    track: Track,
) -> None:
    """Write one manifest per layout under `evaluation/layouts/<name>/`."""
    directory = _set_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    for index, layout in enumerate(layouts):
        manifest = {
            "map": map_name,
            "set": name,
            "index": index,
            "obstacles": [_obstacle_record(item, track) for item in layout],
        }
        path = directory / f"k{index}.json"
        with path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
