"""Filter stack as seen from Python, shared by evaluation and the car.

Track and occupancy paths under
sf/tracks/<map>/, detector/tracker/filter construction, and packing of the
ppo_sf_roboracer message types. Nothing here imports from ppo/, sim/, or
evaluation/; the vehicle needs only vehicle/ and sf/.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import ppo_sf_roboracer as sf
import yaml
from noise import LIDAR_BIAS, LIDAR_STOCHASTIC, ODOM_NOISE_POS_M, OBS_NOISE_YAW

SF_DIR = Path(__file__).resolve().parent
TRACKS = SF_DIR / "tracks"


def track_yaml(map_name: str) -> dict:
    """Read sf/tracks/<map>/track.yaml (map, opt_time, length, direction, start_pose)."""
    with (TRACKS / map_name / "track.yaml").open() as handle:
        return yaml.safe_load(handle)


def load_track(map_name: str) -> sf.TrackHandler:
    """Load raceline.csv, the reference line with its wall clearances."""
    track = sf.TrackHandler()
    path = TRACKS / map_name / "raceline.csv"
    if not track.load(str(path)):
        raise RuntimeError(f"failed to load {path}")
    return track


def occupancy(map_name: str) -> tuple[str, float, float, float]:
    """Occupancy image path, resolution, and origin xy from clean/ else raw/."""
    root = TRACKS / map_name
    folder = root / "clean" if (root / "clean").is_dir() else root / "raw"
    with (folder / f"{map_name}.yaml").open() as handle:
        spec = yaml.safe_load(handle)
    origin = spec["origin"]
    image = folder / str(spec.get("image", f"{map_name}.pgm"))
    return str(image), float(spec["resolution"]), float(origin[0]), float(origin[1])


def wall_points(map_name: str) -> list:
    """boundary.csv as Point2D wall samples."""
    data = np.loadtxt(TRACKS / map_name / "boundary.csv", delimiter=",", skiprows=1)
    points = []
    for x, y in data:
        point = sf.Point2D()
        point.x, point.y = float(x), float(y)
        points.append(point)
    return points


def detector_config(sim2real: bool) -> sf.DetectorConfig:
    """Default detector knobs, widened by the sensor noise when sim2real is on."""
    config = sf.DetectorConfig()
    if sim2real:
        config.lidar_stochastic_error = LIDAR_STOCHASTIC
        config.eps_wall = (
            0.08
            + LIDAR_BIAS
            + 2.0 * (LIDAR_STOCHASTIC + ODOM_NOISE_POS_M)
        )
        config.wall_yaw_error = 2.0 * OBS_NOISE_YAW * math.pi
    return config


def make_detector(
    map_name: str, config: sf.DetectorConfig, track: sf.TrackHandler
) -> sf.ObstacleDetector:
    """Detector on this map's wall samples with the occupancy image loaded."""
    detector = sf.ObstacleDetector(config, track, sf.BoundaryIndex(wall_points(map_name)))
    image, res, ox, oy = occupancy(map_name)
    if not detector.set_static_map(image, res, ox, oy, config.eps_wall + res):
        raise RuntimeError(f"failed to load occupancy {image}")
    return detector


def perceive(
    detector: sf.ObstacleDetector,
    tracker: sf.ObstacleTracker,
    ranges,
    pose: sf.Pose2D,
) -> tuple[list, list]:
    """Raw detections and tracked rectangles for one scan taken at `pose`."""
    raw = detector.detect(list(map(float, ranges)), pose)
    items = list(raw.items) if raw.valid else []
    return items, list(tracker.update(raw))


def lidar_pose(
    x: float, y: float, yaw: float, v: float, r: float, age: float, dist: float
) -> sf.Pose2D:
    """LiDAR origin at scan time: the CoG pose, `age` seconds older than the
    scan, carried forward at constant speed and yaw rate, plus the mount offset.

    Without this every return is placed where the vehicle was, not where it
    is: behind by v*age along the line of sight and rotated by r*age, which at
    range moves it sideways.
    """
    heading = yaw + 0.5 * r * age
    x += v * age * math.cos(heading)
    y += v * age * math.sin(heading)
    yaw += r * age
    pose = sf.Pose2D()
    pose.x = x + dist * math.cos(yaw)
    pose.y = y + dist * math.sin(yaw)
    pose.yaw = yaw
    return pose


def vehicle_state(
    x: float, y: float, yaw: float, delta: float, v: float, r: float
) -> sf.VehicleState:
    """Pack a measured state for the filter."""
    state = sf.VehicleState()
    state.x, state.y, state.yaw, state.delta, state.v, state.r = x, y, yaw, delta, v, r
    return state


def action(delta: float, speed: float) -> sf.Action:
    """Pack a steer/speed command."""
    item = sf.Action()
    item.delta, item.v = delta, speed
    return item
