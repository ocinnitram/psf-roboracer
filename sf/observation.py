"""Policy I/O: downsampled LiDAR beam table, observation vector order, action scaling."""

from __future__ import annotations

import math

import numpy as np

N_ODOM = 4
N_LIDAR = 20
N_ACTION = 2
N_OBS = N_ODOM + N_LIDAR + N_ACTION

# Indices of odometry, LiDAR, and previous-action observation channels.
IDX_S_NORM = 0
IDX_R = 1
IDX_V = 2
IDX_YAW = 3
IDX_LIDAR = 4
IDX_PREV_DELTA = N_ODOM + N_LIDAR
IDX_PREV_V = IDX_PREV_DELTA + 1

# Heading and yaw-rate scales for the odom channels (policy units, not car parameters).
YAW_SCALE_RAD = math.pi
YAW_RATE_SCALE_RADPS = 2.618

# Fractions of the native scan span in units of 1/OBS_BEAM_SCALE. Integer
# arithmetic so Python and the plant land on the same beam index.
OBS_BEAM_SCALE = 10_000
OBS_BEAM_FRAC = (
    0, 972, 1944, 2917,  # -135 to -56 degrees
    3889, 4091, 4293, 4495, 4697, 4899,  # -30 to -3 degrees
    5101, 5303, 5505, 5707, 5909, 6111,  # +3 to +30 degrees
    7083, 8056, 9028, 10000,  # +56 to +135 degrees
)


def obs_beam_indices(n_beams: int) -> np.ndarray:
    """Native-scan indices of the N_LIDAR beams the policy observes."""
    frac = np.asarray(OBS_BEAM_FRAC, dtype=np.int64)
    return (frac * (n_beams - 1) + OBS_BEAM_SCALE // 2) // OBS_BEAM_SCALE


def command_from_action(
    action,
    steer_lock_cmd_rad: float,
    max_speed_mps: float,
):
    """Map policy action in [-1, 1]^2 to steering and speed commands."""
    a_delta = float(action[0])
    a_v = float(action[1])
    return a_delta * steer_lock_cmd_rad, (a_v + 1.0) * 0.5 * max_speed_mps


def action_from_command(
    delta_rad: float,
    speed_mps: float,
    steer_lock_cmd_rad: float,
    max_speed_mps: float,
):
    """Inverse of command_from_action."""
    a_delta = delta_rad / steer_lock_cmd_rad
    a_v = 2.0 * speed_mps / max_speed_mps - 1.0
    return a_delta, a_v
