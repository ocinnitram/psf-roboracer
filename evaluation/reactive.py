"""Reactive baselines driven from the LiDAR scan alone: follow-the-gap and the disparity extender.

Both take the full 811-beam scan (beam 0 at angle_min, angles increase to the left) and return a
steering and speed command. Neither uses the map, the pose or the obstacle tracker.

FollowTheGap ports Evans' FollowTheGap planner and its parameter file from f1tenth_benchmarks
(github.com/BDEvan5/f1tenth_benchmarks, mapless_racing/FollowTheGap.py and params/FollowTheGap.yaml,
commit c973078). That planner runs on the F1TENTH gym scan, 1080 beams over 4.7 rad; its beam-count
parameters are converted here to the same angles on this scan. Its steering is
(index - n/2) * 2*pi/1080 / 2, i.e. the angle of the chosen beam times pi/4.7, which is kept.

DisparityExtender follows Otterness' disparity extender (UNC, 2019 CPSWeek F1/10 race), with the
parameters and speed rule of the UNC code as carried in the Vanderbilt port
(github.com/pmusau17/Platooning-F1Tenth, race/scripts/disparity_extender_vanderbilt.py). The UNC speeds
are motor duty cycles; they are kept as ratios of the top speed.

Both scale their speeds by `speed_scale`, so a sweep traces each method's speed/safety trade-off.

With Evans' parameters follow-the-gap cuts into the inside wall of the first right-hander on the
lab track (about 1.2 m between the walls); his gym maps are wider. `make` therefore widens its
best-point average from 80 to 240 gym beams (FTG_LAB), the only change to his parameters. The
disparity extender runs with the UNC parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Evans' gym scan, which his beam-count parameters are given in.
_EVANS_FOV_RAD = 4.7
_EVANS_BEAMS = 1080
_EVANS_INC_RAD = _EVANS_FOV_RAD / _EVANS_BEAMS


def _beams(n_evans: float, angle_inc: float) -> int:
    """The beam count on this scan covering the angle of n_evans gym beams."""
    return max(1, round(n_evans * _EVANS_INC_RAD / angle_inc))


def _clean(scan: np.ndarray, max_range: float) -> np.ndarray:
    """No-return beams (inf) and NaNs read as the trusted range."""
    out = np.asarray(scan, dtype=np.float64).copy()
    out[~np.isfinite(out)] = max_range
    return np.minimum(out, max_range)


@dataclass
class FollowTheGap:
    """Evans' follow-the-gap (f1tenth_benchmarks), ported by angle to this scan."""

    angle_inc: float
    max_steer: float
    speed_scale: float = 1.0
    # params/FollowTheGap.yaml, in gym beams where it counts beams.
    bubble_radius: float = 160
    preprocess_conv_size: float = 3
    best_point_conv_size: float = 80
    max_lidar_dist: float = 10.0
    fast_speed: float = 5.0
    straights_speed: float = 5.0
    corners_speed: float = 3.0
    straights_steering_angle: float = 0.174
    fast_steering_angle: float = 0.0785
    safe_threshold: float = 5
    # The 135 gym beams dropped from each end of the scan.
    crop: float = 135
    # Multiplies Evans' steering (1 = his pi/4.7 times the beam angle).
    steer_gain: float = 1.0

    def __post_init__(self) -> None:
        inc = self.angle_inc
        self._crop = _beams(self.crop, inc)
        self._bubble = _beams(self.bubble_radius, inc)
        self._pre = max(1, _beams(self.preprocess_conv_size, inc))
        self._best = _beams(self.best_point_conv_size, inc)
        self._safe = _beams(self.safe_threshold, inc)
        # Evans' get_angle uses 2*pi/1080 per gym beam, then halves it.
        self._steer_gain = self.steer_gain * 0.5 * (2.0 * math.pi / _EVANS_BEAMS) / _EVANS_INC_RAD

    def preprocess(self, scan: np.ndarray) -> np.ndarray:
        ranges = _clean(scan, self.max_lidar_dist)[self._crop : -self._crop]
        ranges = np.convolve(ranges, np.ones(self._pre), "same") / self._pre
        return np.clip(ranges, 0.0, self.max_lidar_dist)

    def find_gap(self, ranges: np.ndarray) -> tuple[int, int]:
        """The left-most free run longer than safe_threshold (Evans scans the slices from the end);
        the longest run when none is long enough, where Evans' planner returns nothing."""
        masked = np.ma.masked_where(ranges == 0, ranges)
        slices = np.ma.notmasked_contiguous(masked)
        if not slices:
            return 0, len(ranges)
        for sl in slices[::-1]:
            if sl.stop - sl.start > self._safe:
                return sl.start, sl.stop
        best = max(slices, key=lambda sl: sl.stop - sl.start)
        return best.start, best.stop

    def best_point(self, start: int, end: int, ranges: np.ndarray) -> int:
        averaged = np.convolve(ranges[start:end], np.ones(self._best), "same") / self._best
        return int(averaged.argmax()) + start

    def __call__(self, scan: np.ndarray) -> tuple[float, float]:
        ranges = self.preprocess(scan)
        closest = int(ranges.argmin())
        lo = max(closest - self._bubble, 0)
        hi = min(closest + self._bubble, len(ranges) - 1)
        ranges[lo:hi] = 0.0
        start, end = self.find_gap(ranges)
        best = self.best_point(start, end, ranges)
        beam_angle = (best - len(ranges) / 2.0) * self.angle_inc
        steer = float(np.clip(self._steer_gain * beam_angle, -self.max_steer, self.max_steer))
        if abs(steer) > self.straights_steering_angle:
            speed = self.corners_speed
        elif abs(steer) > self.fast_steering_angle:
            speed = self.straights_speed
        else:
            speed = self.fast_speed
        return steer, self.speed_scale * speed


@dataclass
class DisparityExtender:
    """Otterness' disparity extender, UNC parameters, speeds as ratios of v_top."""

    angle_inc: float
    angle_min: float
    max_steer: float
    v_top: float
    speed_scale: float = 1.0
    car_width: float = 0.5  # "half of the car width, plus some tolerance"
    disparity_threshold: float = 0.3
    turn_clearance: float = 0.3
    max_turn_angle_deg: float = 34.0
    max_range: float = 10.0
    # UNC duty cycles: min 0.37, max 0.5, absolute max 0.6 of the motor.
    min_speed_ratio: float = 0.37 / 0.6
    max_speed_ratio: float = 0.5 / 0.6
    min_distance: float = 0.15
    max_distance: float = 3.0
    no_obstacles_distance: float = 6.0

    def __post_init__(self) -> None:
        n = round(-2.0 * self.angle_min / self.angle_inc) + 1
        self._angles = self.angle_min + self.angle_inc * np.arange(n)
        self._front = np.abs(self._angles) <= 0.5 * math.pi + 1e-9
        self._ahead = int(np.argmin(np.abs(self._angles)))
        # Side windows checked before turning: beyond 90 degrees on each side.
        self._left = self._angles > 0.5 * math.pi
        self._right = self._angles < -0.5 * math.pi

    def extend(self, ranges: np.ndarray) -> np.ndarray:
        """Overwrite, from the nearer side of each disparity, enough beams to cover car_width."""
        out = ranges.copy()
        idx = np.flatnonzero(self._front)
        lo, hi = int(idx[0]), int(idx[-1])
        jumps = np.flatnonzero(np.abs(np.diff(ranges[lo : hi + 1])) >= self.disparity_threshold) + lo
        for i in jumps:
            a, b = ranges[i], ranges[i + 1]
            near, j, step = (a, i, 1) if a < b else (b, i + 1, -1)
            count = math.ceil(self.car_width / max(self.angle_inc * near, 1e-6))
            for _ in range(count):
                if j < lo or j > hi:
                    break
                out[j] = min(out[j], near)
                j += step
        return out

    def speed(self, forward: float) -> float:
        """UNC duty_cycle_from_distance, on the distance straight ahead."""
        v_abs = self.speed_scale * self.v_top
        v_min, v_max = self.min_speed_ratio * v_abs, self.max_speed_ratio * v_abs
        if forward <= self.min_distance:
            return 0.0
        if forward >= self.no_obstacles_distance:
            return v_abs
        if forward >= self.max_distance:
            f = (forward - self.max_distance) / (self.no_obstacles_distance - self.max_distance)
            return v_max + f * (v_abs - v_max)
        f = (forward - self.min_distance) / (self.max_distance - self.min_distance)
        return v_min + f * (v_max - v_min)

    def __call__(self, scan: np.ndarray) -> tuple[float, float]:
        ranges = _clean(scan, self.max_range)
        extended = self.extend(ranges)
        masked = np.where(self._front, extended, -1.0)
        far = np.flatnonzero(masked >= masked.max())
        target = int(far[len(far) // 2])
        max_turn = math.radians(self.max_turn_angle_deg)
        angle = float(np.clip(self._angles[target], -max_turn, max_turn))
        # Don't turn into a wall alongside the car.
        blocked_left = angle > 0.0 and ranges[self._left].min() <= self.turn_clearance
        blocked_right = angle < 0.0 and ranges[self._right].min() <= self.turn_clearance
        if blocked_left or blocked_right:
            angle = 0.0
        steer = float(np.clip(angle, -self.max_steer, self.max_steer))
        return steer, self.speed(float(ranges[self._ahead]))


# The one parameter follow-the-gap changes for the lab track (module docstring).
FTG_LAB = {"best_point_conv_size": 240}


def make(name: str, speed_scale: float, max_steer: float, v_top: float, **params):
    """The named reactive driver for this vehicle's scan."""
    import yaml

    from evaluation.common import REPO

    car = yaml.safe_load((REPO / "vehicle" / "params.yaml").read_text())
    inc = float(car["lidar_angle_inc_rad"])
    if name == "ftg":
        return FollowTheGap(
            angle_inc=inc, max_steer=max_steer, speed_scale=speed_scale, **{**FTG_LAB, **params}
        )
    if name == "disparity":
        return DisparityExtender(
            angle_inc=inc,
            angle_min=float(car["lidar_angle_min_rad"]),
            max_steer=max_steer,
            v_top=v_top,
            speed_scale=speed_scale,
            **params,
        )
    raise ValueError(f"reactive driver must be ftg or disparity, got {name!r}")
