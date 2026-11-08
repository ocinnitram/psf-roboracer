"""Python interface to the Roboracer C simulator."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
_DEFAULT_LIB = REPO / "sim" / "build" / "libroboracer.so"
_DEFAULT_PARAMS = REPO / "vehicle" / "params.yaml"

# Policy yaw-rate unit; same number as YAW_RATE_SCALE_RADPS in sf/observation.py.
_YAW_RATE_SCALE_RADPS = 2.618

DONE_NOT_DONE = 0
COLLISION = 1
COMPLETE = 2

MAX_DYNAMIC_OBSTACLES = 16
OBSTACLE_PACK_FLOATS = 8
_OBSTACLE_INPUT_FIELDS = ("s", "n", "v_s", "half_length", "half_width")
_OBSTACLE_OUTPUT_FIELDS = (
    "cx",
    "cy",
    "yaw",
    "half_length",
    "half_width",
    "vx",
    "vy",
    "s",
)
_MAX_DET_BOXES = 16  # keep equal to MAX_DET_BOXES in renderer.c
_MAX_PLAN = 40

_lib: ctypes.CDLL | None = None

C_VOID_P = ctypes.c_void_p
C_INT = ctypes.c_int
C_FLOAT = ctypes.c_float
C_FLOAT_P = ctypes.POINTER(C_FLOAT)
C_CHAR_P = ctypes.c_char_p
C_DOUBLE = ctypes.c_double
C_UINT = ctypes.c_uint
C_INT8 = ctypes.c_int8

# C initialization key -> vehicle parameter key.
_DIRECT_VEHICLE_PARAMS = {
    "m": "mass_kg",
    "i": "inertia_kgm2",
    "lf": "length_front_m",
    "lr": "length_rear_m",
    "h": "cog_height_m",
    "ego_l": "ego_length_m",
    "ego_w": "ego_width_m",
    "mu": "tyre_mu",
    "c_sf": "cornering_stiffness_front",
    "c_sr": "cornering_stiffness_rear",
    "v_switch": "v_switch_mps",
    "tyre_s": "tyre_s",
    "alpha_pk": "alpha_peak_rad",
    "tau_v": "tau_speed_plant_s",
    "tau_a": "tau_accel_s",
    "a_max": "max_accel_mps2",
    "t_vd": "speed_transport_s",
    "t_sd": "steer_transport_s",
    "tau_delta": "tau_delta_s",
    "tau_delta_dot": "tau_delta_dot_s",
    "steer_gamma": "steer_gamma",
    "delta_th": "steer_delta_th_rad",
    "v_max": "max_speed_mps",
    "lidar_num_scans": "lidar_num_beams",
    "lidar_fov": "lidar_fov_rad",
    "lidar_max_range": "lidar_max_range_m",
    "lidar_mount_offset": "lidar_dist_m",
    "gravity": "gravity_mps2",
    "sigma_fy": "sigma_fy_m",
    "sigma_v_floor": "kinematic_floor_mps",
    "tau_beta_kin": "tau_beta_kin_s",
}


def _ticks_ms(seconds: float) -> float:
    """Convert a period in seconds to an integer millisecond tick count."""
    return float(round(float(seconds) * 1000.0))


def car_kwargs(params_yaml: str | Path | None = None) -> dict[str, float]:
    """Translate shared vehicle parameters into C simulator initialization keys."""
    path = Path(params_yaml) if params_yaml is not None else _DEFAULT_PARAMS
    with path.open() as handle:
        params: dict[str, Any] = yaml.safe_load(handle)

    ego_l = float(params["ego_length_m"])
    ego_w = float(params["ego_width_m"])
    lock = float(params["steer_lock_cmd_rad"])
    rate = float(params["steer_rate_max_radps"])

    result = {
        target: float(params[source])
        for target, source in _DIRECT_VEHICLE_PARAMS.items()
    }
    result.update({
        "r_inscribed": ego_w / 2.0,
        "r_circumscribed": math.hypot(ego_l, ego_w) / 2.0,
        "s_min": -lock,
        "s_max": lock,
        "sv_min": -rate,
        "sv_max": rate,
        "v_min": 0.0,
        "r_max": _YAW_RATE_SCALE_RADPS,
        "yaw_max": math.pi,
        "control_period": _ticks_ms(params["control_period_s"]),
        "odom_period": _ticks_ms(params["odom_period_s"]),
        "compute_latency": _ticks_ms(params["compute_latency_s"]),
        "compute_latency_sd": float(params["compute_latency_sd_s"]) * 1000.0,
    })
    return result


def _declare(
    lib: ctypes.CDLL,
    name: str,
    arguments: Sequence[Any] = (),
    result: Any = None,
) -> None:
    """Declare one function in the C simulator ABI."""
    function = getattr(lib, name)
    function.argtypes = list(arguments)
    function.restype = result


def _declare_many(
    lib: ctypes.CDLL,
    names: Sequence[str],
    arguments: Sequence[Any],
    result: Any,
) -> None:
    """Give several C functions the same ctypes signature."""
    for name in names:
        _declare(lib, name, arguments, result)


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Declare ctypes signatures for the simulator ABI."""
    _declare(
        lib,
        "rr_create",
        [C_INT, C_INT, ctypes.POINTER(C_CHAR_P), ctypes.POINTER(C_DOUBLE)],
        C_VOID_P,
    )
    _declare_many(lib, ("rr_reset", "rr_step", "rr_close"), [C_VOID_P], None)
    _declare_many(
        lib,
        ("rr_observations", "rr_actions", "rr_rewards", "rr_terminals"),
        [C_VOID_P],
        C_FLOAT_P,
    )
    _declare_many(lib, ("rr_obs_size", "rr_num_atns", "rr_render_available"), [], C_INT)

    per_environment_int = (
        "rr_current_lap",
        "rr_done_laps",
        "rr_done_kind",
        "rr_pending_control_period_ms",
        "rr_pending_compute_latency_ms",
        "rr_last_compute_latency_ms",
        "rr_n_beams",
    )
    _declare_many(lib, per_environment_int, [C_VOID_P, C_INT], C_INT)
    _declare_many(
        lib,
        ("rr_last_lap_time", "rr_done_lap_time", "rr_odom_age"),
        [C_VOID_P, C_INT],
        C_FLOAT,
    )
    _declare_many(
        lib,
        ("rr_force_terminal", "rr_reset_env", "rr_render"),
        [C_VOID_P, C_INT],
        None,
    )
    _declare_many(
        lib,
        ("rr_vehicle_state", "rr_tyre_params", "rr_measured_state",
         "rr_raw_scan", "rr_measured_scan"),
        [C_VOID_P, C_INT, C_FLOAT_P],
        None,
    )

    _declare(lib, "rr_set_track", [C_VOID_P, C_INT, C_CHAR_P], C_INT8)
    _declare(lib, "rr_seed", [C_VOID_P, C_INT, C_UINT], None)
    _declare(lib, "rr_set_deterministic", [C_VOID_P, C_INT, C_INT], None)
    _declare(lib, "rr_set_lap_cap", [C_VOID_P, C_INT, C_INT], None)
    _declare(
        lib,
        "rr_set_window",
        [C_VOID_P, C_INT, C_INT, C_FLOAT, C_INT],
        None,
    )
    _declare(
        lib,
        "rr_collision_probe_pose",
        [C_VOID_P, C_INT, C_FLOAT, C_FLOAT, C_FLOAT, C_FLOAT_P],
        None,
    )
    _declare(lib, "rr_set_obstacles", [C_VOID_P, C_INT, C_FLOAT_P, C_INT], None)
    _declare(
        lib,
        "rr_get_obstacles",
        [C_VOID_P, C_INT, C_FLOAT_P, ctypes.POINTER(C_INT)],
        None,
    )

    _declare(lib, "rr_set_det_boxes", [C_FLOAT_P, C_INT], None)
    _declare_many(lib, ("rr_set_safety", "rr_set_trace"), [C_INT], None)
    _declare(lib, "rr_push_trace", [C_FLOAT, C_FLOAT, C_INT], None)
    _declare(lib, "rr_set_filter_path", [C_FLOAT_P, C_INT], None)
    return lib


def load_library(path: str | Path | None = None) -> ctypes.CDLL:
    """Load libroboracer.so and cache the handle."""
    global _lib
    if _lib is not None and path is None:
        return _lib
    so_path = Path(path) if path is not None else _DEFAULT_LIB
    if not so_path.is_file():
        raise FileNotFoundError(
            f"{so_path} not found. Build it with `make -C sim`."
        )
    lib = _bind(ctypes.CDLL(str(so_path)))
    if path is None:
        _lib = lib
    return lib


def _as_array(ptr: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Wrap a C float buffer as a numpy view."""
    return np.ctypeslib.as_array(ptr, shape=shape)


# Keys the plant needs on top of car_kwargs(): rewards, noise, and the run's pace target.
RUN_KEYS = (
    "progress_reward",
    "collision_reward",
    "lap_completion_reward",
    "optimal_lap_time",
    "sim2real",
    "obs_noise_s_norm",
    "obs_noise_yaw_rate",
    "obs_noise_vel",
    "obs_noise_yaw",
    "odom_noise_pos_m",
    "odom_noise_vel_mps",
    "lidar_bias",
    "lidar_stochastic",
)

_DERIVED_VEHICLE_KEYS = {
    "r_inscribed",
    "r_circumscribed",
    "s_min",
    "s_max",
    "sv_min",
    "sv_max",
    "v_min",
    "r_max",
    "yaw_max",
    "control_period",
    "odom_period",
    "compute_latency",
}
REQUIRED_INIT_KEYS = frozenset(_DIRECT_VEHICLE_PARAMS) | _DERIVED_VEHICLE_KEYS | set(RUN_KEYS)


def _create_vector(
    lib: ctypes.CDLL,
    num_envs: int,
    kwargs: dict[str, float],
) -> C_VOID_P:
    """Create the C vector environment from named initialization values."""
    items = list(kwargs.items())
    key_storage = [key.encode("utf-8") for key, _ in items]
    keys = (C_CHAR_P * len(items))(*key_storage)
    values = (C_DOUBLE * len(items))(*[float(value) for _, value in items])
    handle = lib.rr_create(num_envs, len(items), keys, values)
    if not handle:
        raise RuntimeError("rr_create failed")
    return handle


class VecEnv:
    """Batched C plant with zero-copy views of obs, actions, rewards, and terminals."""

    def __init__(
        self,
        num_envs: int,
        blob_path: str | Path,
        kwargs: dict[str, float],
        seed: int,
        deterministic: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        track_blob = Path(blob_path)
        if not track_blob.is_file():
            raise FileNotFoundError(f"track blob not found: {track_blob}")

        self._lib = load_library()
        missing = sorted(REQUIRED_INIT_KEYS - set(kwargs))
        if missing:
            raise ValueError(f"VecEnv kwargs missing {missing}; start from car_kwargs()")

        os.environ["ROBORACER_TRACK"] = str(track_blob)
        self._vec = _create_vector(self._lib, self.num_envs, kwargs)
        self._attach_buffers()
        self._seed_environments(seed, deterministic)

    def _attach_buffers(self) -> None:
        """Expose the C-owned buffers as zero-copy NumPy arrays."""
        obs_size = int(self._lib.rr_obs_size())
        n_atns = int(self._lib.rr_num_atns())
        self.obs = _as_array(
            self._lib.rr_observations(self._vec),
            (self.num_envs, obs_size),
        )
        self.actions = _as_array(
            self._lib.rr_actions(self._vec),
            (self.num_envs, n_atns),
        )
        self.rewards = _as_array(
            self._lib.rr_rewards(self._vec),
            (self.num_envs,),
        )
        self.terminals = _as_array(
            self._lib.rr_terminals(self._vec),
            (self.num_envs,),
        )

    def _seed_environments(self, seed: int, deterministic: bool) -> None:
        """Seed each environment independently and configure determinism."""
        for i in range(self.num_envs):
            self._lib.rr_seed(self._vec, i, seed * 1000 + i)
            if deterministic:
                self._lib.rr_set_deterministic(self._vec, i, 1)

    def seed(self, i: int, seed: int) -> None:
        """Set one env's RNG. Call before reset."""
        self._lib.rr_seed(self._vec, i, seed)

    def reset(self) -> np.ndarray:
        """Reset every env and return a copy of the observation buffer."""
        self._lib.rr_reset(self._vec)
        return self.obs.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Write actions, step, and return copies of obs, rewards, and terminals."""
        self.actions[:] = np.asarray(actions, dtype=np.float32)
        self._lib.rr_step(self._vec)
        return self.obs.copy(), self.rewards.copy(), self.terminals.copy()

    def close(self) -> None:
        """Free the C vec and drop the numpy views that alias its buffers."""
        if getattr(self, "_vec", None) is None:
            return
        del self.obs, self.actions, self.rewards, self.terminals
        self._lib.rr_close(self._vec)
        self._vec = None

    def set_track(self, i: int, blob_path: str | Path) -> int:
        """Swap the mmap'd track blob for one env. Returns 0 on success, -1 on error."""
        return int(self._lib.rr_set_track(self._vec, i, str(blob_path).encode()))

    def measured_state(self, i: int) -> np.ndarray:
        """Return deploy pose [x, y, yaw, delta, v, yaw_rate] for env i."""
        out = np.empty(6, dtype=np.float32)
        self._lib.rr_measured_state(
            self._vec, i, out.ctypes.data_as(C_FLOAT_P),
        )
        return out

    def vehicle_state(self, i: int) -> np.ndarray:
        """Return the true plant state [x, y, yaw, delta, v, r, beta, accel,
        delta_dot, fy_f, fy_r] for env i (raw yaw)."""
        out = np.empty(11, dtype=np.float32)
        self._lib.rr_vehicle_state(self._vec, i, out.ctypes.data_as(C_FLOAT_P))
        return out

    def odom_age(self, i: int) -> float:
        """Return how old measured_state is at the current scan, in seconds."""
        return float(self._lib.rr_odom_age(self._vec, i))

    def raw_scan(self, i: int, out: np.ndarray | None = None) -> np.ndarray:
        """Write the full-resolution noiseless scan in metres into out (or a new array)."""
        n = int(self._lib.rr_n_beams(self._vec, i))
        buf = out if out is not None else np.empty(n, dtype=np.float32)
        self._lib.rr_raw_scan(self._vec, i, buf.ctypes.data_as(C_FLOAT_P))
        return buf

    def measured_scan(self, i: int, out: np.ndarray | None = None) -> np.ndarray:
        """Write the full-resolution measured scan in metres into out (or a new array)."""
        n = int(self._lib.rr_n_beams(self._vec, i))
        buf = out if out is not None else np.empty(n, dtype=np.float32)
        self._lib.rr_measured_scan(self._vec, i, buf.ctypes.data_as(C_FLOAT_P))
        return buf

    def set_obstacles(self, i: int, obstacles: list[dict[str, float]]) -> None:
        """Install track-frame rectangles on env i after reset; refreshes policy obs."""
        n = len(obstacles)
        if n > MAX_DYNAMIC_OBSTACLES:
            raise ValueError(
                f"{n} obstacles exceeds MAX_DYNAMIC_OBSTACLES={MAX_DYNAMIC_OBSTACLES}"
            )
        buf = (ctypes.c_float * (MAX_DYNAMIC_OBSTACLES * OBSTACLE_PACK_FLOATS))()
        for j, item in enumerate(obstacles):
            for k, key in enumerate(_OBSTACLE_INPUT_FIELDS):
                buf[j * OBSTACLE_PACK_FLOATS + k] = float(item[key])
        self._lib.rr_set_obstacles(self._vec, i, buf, n)

    def get_obstacles(self, i: int) -> list[dict[str, float]]:
        """Read live obstacles as world pose, size, velocity, and station."""
        buf = (ctypes.c_float * (MAX_DYNAMIC_OBSTACLES * OBSTACLE_PACK_FLOATS))()
        n = ctypes.c_int(0)
        self._lib.rr_get_obstacles(self._vec, i, buf, ctypes.byref(n))
        rows: list[dict[str, float]] = []
        for j in range(n.value):
            base = j * OBSTACLE_PACK_FLOATS
            rows.append({
                key: float(buf[base + k])
                for k, key in enumerate(_OBSTACLE_OUTPUT_FIELDS)
            })
        return rows

    def set_window(
        self,
        i: int,
        start_ndx: int,
        end_s_norm: float,
        tick_cap: int = 0,
    ) -> None:
        """Spawn env i on the racing line at start_ndx and complete at end_s_norm."""
        self._lib.rr_set_window(self._vec, i, start_ndx, float(end_s_norm), tick_cap)

    def set_lap_cap(self, i: int, laps: int) -> None:
        """Set how many laps env i must finish before COMPLETE."""
        self._lib.rr_set_lap_cap(self._vec, i, laps)

    def set_tick_cap(self, i: int, ticks: int) -> None:
        """Cap env i's episodes at `ticks` plant ticks (0 = the built-in MAX_TICKS)."""
        self._lib.rr_set_window(self._vec, i, 0, -1.0, int(ticks))

    def force_terminal(self, i: int) -> None:
        """Force env i to terminate and reset inside C."""
        self._lib.rr_force_terminal(self._vec, i)

    def reset_env(self, i: int) -> None:
        """Reset one env without latching a terminal."""
        self._lib.rr_reset_env(self._vec, i)

    def done_laps(self, i: int) -> int:
        """Return the lap count latched at the last terminal."""
        return int(self._lib.rr_done_laps(self._vec, i))

    def done_lap_time(self, i: int) -> float:
        """Return the flying-lap time latched at the last terminal."""
        return float(self._lib.rr_done_lap_time(self._vec, i))

    def done_kind(self, i: int) -> int:
        """Return DONE_NOT_DONE, COLLISION, or COMPLETE latched at the last terminal."""
        return int(self._lib.rr_done_kind(self._vec, i))

    def render(self, i: int) -> None:
        """Draw env i in the native window."""
        self._lib.rr_render(self._vec, i)


class Renderer:
    """Live window overlays; no-ops when the library was built without raylib."""

    def __init__(self) -> None:
        self._lib = load_library()
        self._ok = int(self._lib.rr_render_available()) != 0
        if not self._ok:
            print(
                "Renderer: rr_render_available() is 0; "
                "rebuild with `make -C sim RENDER=1`.",
                file=sys.stderr,
            )

    def render(self, venv: VecEnv, i: int) -> None:
        """Draw env i, or do nothing when raylib is not linked."""
        if self._ok:
            venv.render(i)

    def set_det_boxes(self, boxes: np.ndarray) -> None:
        """Push detector boxes as packed [x, y, yaw, length, width] metres."""
        if not self._ok:
            return
        arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
        n = min(arr.shape[0], _MAX_DET_BOXES)
        buf = (ctypes.c_float * (_MAX_DET_BOXES * 5))()
        for i in range(n):
            for k in range(5):
                buf[i * 5 + k] = float(arr[i, k])
        self._lib.rr_set_det_boxes(buf, n)

    def set_safety(self, mode: int) -> None:
        """Set the safety chip: 0 clear, 1 filter, 2 unsafe."""
        if self._ok:
            self._lib.rr_set_safety(int(mode))

    def set_trace(self, on: bool) -> None:
        """Enable or clear the pose trace overlay."""
        if self._ok:
            self._lib.rr_set_trace(1 if on else 0)

    def push_trace(self, x: float, y: float, mode: int) -> None:
        """Append one trace sample at (x, y)."""
        if self._ok:
            self._lib.rr_push_trace(float(x), float(y), int(mode))

    def set_filter_path(self, points: np.ndarray) -> None:
        """Push the predicted horizon as packed [x, y] metres."""
        if not self._ok:
            return
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        n = min(arr.shape[0], _MAX_PLAN)
        buf = (ctypes.c_float * (_MAX_PLAN * 2))()
        for i in range(n):
            buf[i * 2 + 0] = float(arr[i, 0])
            buf[i * 2 + 1] = float(arr[i, 1])
        self._lib.rr_set_filter_path(buf, n)
