"""car_kwargs maps vehicle/params.yaml onto the C my_init car keys."""

from __future__ import annotations

import math

import pytest

import roboracer_sim as rr

CAR_KEYS = {
    "m", "i", "lf", "lr", "h", "ego_w", "ego_l", "r_inscribed", "r_circumscribed",
    "mu", "c_sf", "c_sr", "v_switch", "tyre_s", "alpha_pk", "tau_v", "tau_a",
    "a_max", "t_vd", "t_sd", "tau_delta", "tau_delta_dot", "steer_gamma",
    "delta_th", "s_min", "s_max", "sv_min", "sv_max", "v_max", "v_min", "r_max",
    "yaw_max", "lidar_num_scans", "lidar_fov", "lidar_max_range",
    "lidar_mount_offset", "control_period", "odom_period", "compute_latency",
    "compute_latency_sd", "gravity", "sigma_fy", "sigma_v_floor", "tau_beta_kin",
}

# Values computed from params.yaml rather than copied from it.
DERIVED = {
    "r_inscribed": 0.145,
    "r_circumscribed": 0.2890069203323685,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_min": 0.0,
    "r_max": 2.618,
    "yaw_max": math.pi,
    "control_period": 67.0,
    "odom_period": 20.0,
    "compute_latency": 20.0,
    "compute_latency_sd": 0.0,
}


def test_car_kwargs_keys() -> None:
    assert set(rr.car_kwargs()) == CAR_KEYS


def test_car_kwargs_derived_values() -> None:
    got = rr.car_kwargs()
    for key, expected in DERIVED.items():
        assert got[key] == pytest.approx(expected, abs=1e-6, rel=0.0), key
