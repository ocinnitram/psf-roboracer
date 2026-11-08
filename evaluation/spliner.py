"""ForzaETH's spliner: evasion splines around tracked obstacles, followed by the MAP controller.

A port of the head-to-head stack of the ForzaETH race stack (Baumann et al., "ForzaETH Race Stack -
Scaled Autonomous Head-to-Head Racing on Fully Commercial Off-the-Shelf Hardware", JFR 2024;
github.com/ForzaETH/race_stack, commit 67981ed): the spline planner
(planner/local_planners/spline_planner/spline_planner.py), the head-to-head state machine
(state_machine/state_machine.py, transitions.py) and the MAP controller (model- and acceleration-based
pursuit, controller/controller/map.py) with its trailing controller, with the SIM parameters of
stack_master/config/SIM/l1_params.yaml and stack_master/config/state_machine_params.yaml. Pure pursuit
(controller/controller/pp.py) is kept as `make(..., controller="pp")`.

MAP steers like pure pursuit to the point L1 ahead, but turns the lateral acceleration of that arc,
2 v^2 sin(eta) / L1 at the path's speed, into a steering command through a lookup table of the vehicle's
steady-state cornering (steering_lookup/lookup_steer_angle.py). The stack builds that table from its
system identification; here it is evaluation/data/map_lut.csv, written by evaluation/tools/map_lut.cpp
from this vehicle's identified model: the steering transmission (gamma curve) and the filter's
steady-state Pacejka single-track cornering (sf/src/sf_model.cpp), the same model the filter predicts with.

The global trajectory is the track's raceline (raceline.csv) with a forward-backward speed profile in the
vehicle's acceleration limits (lat_accel_hi_mps2, max_accel_mps2; `Line.speed_profile`). The stack would
use a minimum-time line, but this track's (raceline_mintime.csv) runs within 0.17 m of the walls, and
pure pursuit cutting its corners hits them at every speed scale tried; the raceline keeps at least
0.41 m, as the stack's global planner keeps its line a safety width from the walls. `speed_scale`
scales the profile like the stack's velocity scaler. Obstacles are the tracker's boxes, converted to
Frenet coordinates of the line.

Lab-track parameters (everything else is the stack's):
  - m_l1 0.6 for MAP, 0.7 for pure pursuit (SIM 0.3). With 0.3 the car oscillates on the line under the
    vehicle's steering latency and hits the walls of the empty track. q_l1 and t_clip_min stay at the
    stack's values. Chosen on the tuning layouts (evaluation/layouts/obs-set-tune-50, apart from the
    held-out set), where MAP laps the empty track at speed scales 0.6, 0.8 and 1.0 on all 3 plant seeds.
  - obs_traj_tresh and lateral_width_gb both 0.4 m (stack 0.3 and 1.1). With the stack's pair a box
    between 0.3 and 1.1 m off the line is not splined around but still blocks the line, so the car
    trails it to a stop; one width for both removes that.
  - evasion_dist 0.4 m (stack 0.65, sized for an opponent car), which completes more of the tuning
    layouts at speed scale 0.8 than 0.65 does.

Each period:
  1. the spliner takes the closest obstacle within `lookahead` ahead that lies within `obs_traj_tresh` of
     the line, places an apex `evasion_dist` beside it on the side with more room, and fits a spline in
     d(s) through seven points (pre-apex, apex, post-apex) spread with the speed; a spline that comes
     within `spline_bound_mindist` of a wall, or switches sides off the line, is rejected;
  2. the state machine picks GB_TRACK (the line is free within `gb_horizon`), OVERTAKE (a valid
     spline that clears every obstacle by `lateral_width_ot`) or TRAILING (the line, at the speed of the
     trailing controller, which holds `trailing_gap` behind the obstacle);
  3. MAP (or pure pursuit) steers to the point L1 = q_l1 + m_l1 v ahead on the chosen path and commands
     the path's speed.

Left out, as in a race with static obstacles they do not apply: the obstacle-motion prediction (the
boxes are static), the overtaking sectors (overtaking is allowed everywhere), the low-battery and
FTG-only states, and the steer scaling on acceleration and above 7 m/s (both at 1.0 in the SIM
parameters, and this car tops out at 4 m/s). The stack's FTGONLY state (follow-the-gap after 3 s
standing still in TRAILING) is off in the stack itself (state_machine.py sets ftg_disabled = True, and
the controller manager in MAP mode never runs follow-the-gap), so a car that cannot pass a box trails it
to a stop, as in the stack, and is counted as parked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

GB_TRACK, TRAILING, OVERTAKE = "GB_TRACK", "TRAILING", "OVERTAKE"
WPNT_DIST = 0.1  # the stack's waypoint and spline resolution [m]


class Line:
    """A closed reference line with Frenet conversions (left of the line is +d)."""

    def __init__(self, csv_path) -> None:
        d = np.genfromtxt(csv_path, delimiter=",", names=True)
        x, y, s = d["x"], d["y"], d["s"]
        self.length = float(s[-1] + math.hypot(x[0] - x[-1], y[0] - y[-1]))
        knots = np.append(s, self.length)
        self._x = CubicSpline(knots, np.append(x, x[0]), bc_type="periodic")
        self._y = CubicSpline(knots, np.append(y, y[0]), bc_type="periodic")
        # Resampled at the stack's waypoint spacing.
        self.s = np.arange(0.0, self.length, WPNT_DIST)
        self.x, self.y = self._x(self.s), self._y(self.s)
        tx, ty = self._x(self.s, 1), self._y(self.s, 1)
        norm = np.hypot(tx, ty)
        self.tx, self.ty = tx / norm, ty / norm
        self.yaw = np.arctan2(self.ty, self.tx)
        per = lambda col: np.interp(self.s, knots, np.append(d[col], d[col][0]))
        self.kappa, self.w_l, self.w_r = per("kappa"), per("w_l"), per("w_r")
        self.v_ref = per("v_ref") if "v_ref" in d.dtype.names else None

    def speed_profile(self, v_max: float, a_lat: float, a_x: float) -> np.ndarray:
        """Forward-backward speed profile on the curvature, inside a friction ellipse."""
        n, ds = len(self.s), WPNT_DIST
        v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(self.kappa), 1e-6)))
        for _ in range(2):  # twice round the loop so the wrap converges
            for step in (1, -1):
                order = range(n) if step == 1 else range(n - 1, -1, -1)
                for i in order:
                    j = (i - step) % n
                    ay = v[j] ** 2 * abs(self.kappa[j])
                    ax = a_x * math.sqrt(max(0.0, 1.0 - (ay / a_lat) ** 2))
                    v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * ax * ds))
        return v

    def index(self, s) -> np.ndarray | int:
        return np.round(np.asarray(s) / WPNT_DIST).astype(int) % len(self.s)

    def to_frenet(self, px, py) -> tuple[np.ndarray, np.ndarray]:
        """Project points onto the line: the nearest waypoint, then onto its tangent."""
        px, py = np.atleast_1d(px).astype(float), np.atleast_1d(py).astype(float)
        i = np.argmin((self.x[None, :] - px[:, None]) ** 2 + (self.y[None, :] - py[:, None]) ** 2, axis=1)
        dx, dy = px - self.x[i], py - self.y[i]
        s = (self.s[i] + dx * self.tx[i] + dy * self.ty[i]) % self.length
        # One refinement on the spline itself.
        cx, cy = self._x(s), self._y(s)
        tx, ty = self._x(s, 1), self._y(s, 1)
        tn = np.hypot(tx, ty)
        tx, ty = tx / tn, ty / tn
        dx, dy = px - cx, py - cy
        s = (s + dx * tx + dy * ty) % self.length
        cx, cy = self._x(s), self._y(s)
        tx, ty = self._x(s, 1), self._y(s, 1)
        tn = np.hypot(tx, ty)
        d = (tx * (py - cy) - ty * (px - cx)) / tn
        return s, d

    def to_cartesian(self, s, d) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float) % self.length
        tx, ty = self._x(s, 1), self._y(s, 1)
        tn = np.hypot(tx, ty)
        return self._x(s) - d * ty / tn, self._y(s) + d * tx / tn


@dataclass
class FrenetObstacle:
    s_start: float
    s_end: float
    s_center: float
    d_left: float
    d_right: float
    d_center: float


def frenet_obstacle(line: Line, cx: float, cy: float, yaw: float, length: float, width: float) -> FrenetObstacle:
    """A tracked box in Frenet coordinates: the extent of its corners along and across the line."""
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[a * length / 2, b * width / 2] for a in (-1, 1) for b in (-1, 1)])
    px = cx + loc[:, 0] * c - loc[:, 1] * s
    py = cy + loc[:, 0] * s + loc[:, 1] * c
    s_c, d_c = line.to_frenet(cx, cy)
    s_k, d_k = line.to_frenet(px, py)
    half = line.length / 2
    ds = (s_k - s_c[0] + half) % line.length - half  # corners relative to the centre, unwrapped
    return FrenetObstacle(
        s_start=float((s_c[0] + ds.min()) % line.length),
        s_end=float((s_c[0] + ds.max()) % line.length),
        s_center=float(s_c[0]),
        d_left=float(d_k.max()),
        d_right=float(d_k.min()),
        d_center=float(d_c[0]),
    )


class SteerLookup:
    """The stack's steering lookup (steering_lookup/lookup_steer_angle.py): the steering command whose
    steady-state lateral acceleration is a_y, at the nearest tabulated speed, interpolated between the
    two nearest tabulated accelerations. Table: first row 0 and speeds, first column steering commands,
    entries a_y (evaluation/data/map_lut.csv, from evaluation/tools/map_lut.cpp)."""

    def __init__(self, csv_path) -> None:
        lu = np.loadtxt(csv_path, delimiter=",")
        self.v, self.steer, self.a_y = lu[0, 1:], lu[1:, 0], lu[1:, 1:]

    def __call__(self, a_y: float, v: float) -> float:
        sign = 1.0 if a_y > 0.0 else -1.0
        a = abs(a_y)
        col = self.a_y[:, int(np.abs(self.v - v).argmin())]
        i = int(np.abs(col - a).argmin())
        if i in (0, len(col) - 1):
            return sign * float(self.steer[i])
        nb = (i - 1, i + 1)
        j = nb[int(np.abs(col[list(nb)] - a).argmin())]
        if col[j] == col[i]:
            return sign * float(self.steer[min(i, j)])
        return sign * float(np.interp(a, sorted((col[i], col[j])), [self.steer[min(i, j, key=lambda q: col[q])],
                                                                    self.steer[max(i, j, key=lambda q: col[q])]]))


@dataclass
class Evasion:
    s: np.ndarray
    d: np.ndarray
    v: np.ndarray
    side: str


@dataclass
class Spliner:
    """The stack's spline planner, state machine and controller (MAP; pure pursuit when `lut` is None)."""

    line: Line
    max_steer: float
    wheelbase: float
    v_max: float
    speed_scale: float = 1.0
    dt: float = 0.067
    # Spline planner (spline_planner.py defaults).
    pre_apex: tuple[float, float, float] = (-4.0, -3.0, -1.5)
    post_apex: tuple[float, float, float] = (2.0, 3.0, 4.0)
    evasion_dist: float = 0.4  # 0.65 published; see the module docstring
    obs_traj_tresh: float = 0.4  # 0.3 published
    spline_bound_mindist: float = 0.2
    lookahead: float = 10.0
    outside_stretch: float = 1.75
    inside_speed: float = 0.9
    side_switch_d: float = 0.25
    # State machine (state_machine_params.yaml).
    splini_ttl: float = 2.0
    gb_ego_width: float = 0.4
    lateral_width_gb: float = 0.4  # 1.1 published; = obs_traj_tresh
    gb_horizon: float = 6.9
    overtaking_horizon: float = 6.9
    lateral_width_ot: float = 0.3
    splini_hyst_timer: float = 0.2
    # Pure pursuit and trailing (SIM/l1_params.yaml).
    t_clip_min: float = 1.0
    t_clip_max: float = 5.0
    m_l1: float = 0.7  # SIM 0.3; see the module docstring
    q_l1: float = 0.15
    trailing_gap: float = 1.5
    trailing_p_gain: float = 0.5
    trailing_i_gain: float = 0.001
    trailing_d_gain: float = 0.2
    steer_step_max: float = 0.4
    # Speed profile of the global line (the vehicle's lat_accel_hi_mps2 and max_accel_mps2).
    a_lat: float = 4.94
    a_x: float = 3.48

    # MAP's steering lookup (SteerLookup); None steers by pure pursuit.
    lut: SteerLookup | None = None

    state: str = field(default=GB_TRACK, init=False)
    rejected: str = field(default="", init=False)  # why this period's spline was rejected
    _last_side: str = field(default="", init=False)
    _last_switch_t: float = field(default=-1e9, init=False)
    _avoid: Evasion | None = field(default=None, init=False)
    _avoid_t: float = field(default=-1e9, init=False)
    _valid_avoid: Evasion | None = field(default=None, init=False)
    _t: float = field(default=0.0, init=False)
    _steer: float = field(default=0.0, init=False)
    _i_gap: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        v = self.line.speed_profile(self.v_max, self.a_lat, self.a_x)
        self.v_gb = np.minimum(self.speed_scale * v, self.v_max)
        self.v_top = float(self.v_gb.max())

    # ---- spline planner ----
    def _more_space(self, ob: FrenetObstacle, i: int) -> tuple[str, float]:
        left_gap = abs(self.line.w_l[i] - ob.d_left)
        right_gap = abs(self.line.w_r[i] + ob.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist
        if right_gap > min_space and left_gap < min_space:
            return "right", min(ob.d_right - self.evasion_dist, 0.0)
        if left_gap > min_space and right_gap < min_space:
            return "left", max(ob.d_left + self.evasion_dist, 0.0)
        left = ob.d_left + self.evasion_dist
        right = ob.d_right - self.evasion_dist
        if abs(left) <= abs(right):
            return "left", max(left, 0.0)
        return "right", min(right, 0.0)

    def _close(self, obstacles: list[FrenetObstacle], s: float) -> list[FrenetObstacle]:
        L = self.line.length
        return [
            ob
            for ob in obstacles
            if abs(ob.d_center) < self.obs_traj_tresh and (ob.s_center - s) % L < self.lookahead
        ]

    def plan(self, obstacles: list[FrenetObstacle], s: float, d: float, vs: float) -> Evasion | None:
        """The evasion spline around the closest obstacle, or None (no obstacle, or rejected)."""
        self.rejected = ""
        close = self._close(obstacles, s)
        if not close:
            return None
        line, L = self.line, self.line.length
        ob = min(close, key=lambda o: (o.s_center - s) % L)
        s_end = ob.s_end + L if ob.s_end < ob.s_start else ob.s_end
        s_apex = (ob.s_start + s_end) / 2
        idx = [int(s_apex / WPNT_DIST + k) % len(line.s) for k in range(20)]
        outside = "left" if np.sum(line.kappa[idx]) < 0 else "right"
        side, d_apex = self._more_space(ob, idx[0])
        points = []
        for dst in (*self.pre_apex, 0.0, *self.post_apex):
            dst = dst * float(np.clip(1.0 + vs / self.v_top, 1.0, 1.5))
            si = s_apex + dst * (self.outside_stretch if outside == side else 1.0)
            points.append((si, d_apex if dst == 0 else 0.0))
        pts = np.array(points)
        ev_s = np.arange(pts[0, 0], pts[-1, 0], WPNT_DIST)
        ev_d = Spline(pts[:, 0], pts[:, 1])(ev_s)
        ev_d = np.clip(ev_d, d_apex, 0.0) if d_apex < 0 else np.clip(ev_d, 0.0, d_apex)
        ev_s = ev_s % L
        if abs(d) > self.side_switch_d and side != self._last_side:
            self.rejected = "side_switch"
        i = line.index(ev_s)
        bound = line.w_l[i] if side == "left" else line.w_r[i]
        off = np.abs(ev_d) > WPNT_DIST
        if np.any(off & (np.abs(ev_d) > np.abs(bound) - self.spline_bound_mindist)):
            self.rejected = "bounds"
        if self.rejected:
            # The stack's "cool down": a rejected spline counts as a side switch.
            self._last_switch_t = self._t
            return None
        if side != self._last_side:
            self._last_switch_t = self._t
        self._last_side = side
        v = self.v_gb[i] * (1.0 if outside == side else self.inside_speed)
        return Evasion(ev_s, ev_d, v, side)

    # ---- state machine ----
    def _gb_free(self, obstacles, s) -> bool:
        L = self.line.length
        return not any(
            (ob.s_center - s) % L < self.gb_horizon and abs(ob.d_center) < self.lateral_width_gb
            for ob in obstacles
        )

    def _splini_available(self) -> bool:
        if self._avoid is None or self._t - self._avoid_t > self.splini_ttl:
            return False
        if abs(self._avoid_t - self._last_switch_t) < self.splini_hyst_timer:
            return False
        self._valid_avoid = self._avoid
        return True

    def _o_free(self, obstacles, s) -> bool:
        ev = self._valid_avoid
        if ev is None:
            return True
        L = self.line.length
        for ob in obstacles:
            if (ob.s_center - s) % L < self.overtaking_horizon:
                j = int(np.argmin(np.abs(ev.s - ob.s_center)))
                if abs(ev.d[j] - ob.d_center) < self.lateral_width_ot:
                    return False
        return True

    def _transition(self, obstacles, s, d) -> str:
        gb_free = self._gb_free(obstacles, s)
        if self.state == GB_TRACK:
            return GB_TRACK if gb_free else TRAILING
        if self.state == TRAILING:
            if gb_free and abs(d) < self.gb_ego_width:
                return GB_TRACK
            if not gb_free and self._splini_available() and self._o_free(obstacles, s):
                return OVERTAKE
            return TRAILING
        # OVERTAKE (overtaking is allowed everywhere: always in an OT sector)
        if not self._o_free(obstacles, s):
            return TRAILING
        if self._splini_available():
            return OVERTAKE
        return GB_TRACK if gb_free else TRAILING

    # ---- controller ----
    def _path(self, s: float, overtake: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Waypoints (x, y, v) of the path ahead: the line, with the evasion spline spliced in."""
        line = self.line
        ss = s + np.arange(-1.0, self.t_clip_max + 2.0, WPNT_DIST)
        i = line.index(ss)
        dd = np.zeros_like(ss)
        vv = self.v_gb[i].copy()
        ev = self._valid_avoid if overtake else None
        if ev is not None:
            L = line.length
            rel = (ev.s - ev.s[0]) % L
            span = rel[-1]
            pos = (ss - ev.s[0]) % L
            inside = pos <= span
            j = np.clip(np.searchsorted(rel, pos[inside]), 0, len(rel) - 1)
            dd[inside], vv[inside] = ev.d[j], ev.v[j]
        x, y = line.to_cartesian(ss, dd)
        return np.c_[x, y], dd, vv

    def _trailing_speed(self, obstacles, s, vs, v_gb) -> float:
        L = self.line.length
        ahead = [ob for ob in obstacles if (ob.s_center - s) % L < self.gb_horizon]
        if not ahead:
            return v_gb
        ob = min(ahead, key=lambda o: (o.s_center - s) % L)
        gap = (ob.s_center - s) % L
        err = self.trailing_gap - gap
        self._i_gap = float(np.clip(self._i_gap + err * self.dt, -10, 10))
        cmd = -self.trailing_p_gain * err - self.trailing_i_gain * self._i_gap - self.trailing_d_gain * vs
        return float(np.clip(cmd, 0.0, v_gb))

    def __call__(self, x: float, y: float, yaw: float, v: float, obstacles: list[FrenetObstacle]):
        """(steer, speed) for this period."""
        s_arr, d_arr = self.line.to_frenet(x, y)
        s, d = float(s_arr[0]), float(d_arr[0])
        vs = v * math.cos(yaw - float(self.line.yaw[self.line.index(s)]))
        ev = self.plan(obstacles, s, d, vs)
        if ev is not None:
            self._avoid, self._avoid_t = ev, self._t
        self.state = self._transition(obstacles, s, d)
        pts, _, vv = self._path(s, self.state == OVERTAKE)
        k = int(np.argmin(np.hypot(pts[:, 0] - x, pts[:, 1] - y)))
        # The stack bounds L1 below by the lateral error to the global line, also while overtaking.
        l1 = float(np.clip(self.q_l1 + v * self.m_l1, max(self.t_clip_min, math.sqrt(2) * abs(d)), self.t_clip_max))
        target = pts[min(k + int(l1 / WPNT_DIST + 0.5), len(pts) - 1)]
        vec = target - (x, y)
        n = float(np.hypot(*vec))
        eta = 0.0 if n == 0 else math.asin(float(np.dot((-math.sin(yaw), math.cos(yaw)), vec)) / n)
        if self.lut is None:  # pure pursuit (pp.py)
            steer = math.atan(2 * self.wheelbase * math.sin(eta) / l1)
        else:  # MAP (map.py): the lateral acceleration of the L1 arc at the path's speed, through the LUT
            v_lu = float(vv[k])
            steer = self.lut(2 * v_lu**2 / l1 * math.sin(eta), v_lu)
        steer *= float(np.clip(1 + v / 10, 1, 1.25))
        steer = float(np.clip(steer, self._steer - self.steer_step_max, self._steer + self.steer_step_max))
        self._steer = steer
        speed = float(vv[k])
        if self.state == TRAILING:
            speed = self._trailing_speed(obstacles, s, vs, speed)
        else:
            self._i_gap = 0.0
        self._t += self.dt
        return float(np.clip(steer, -self.max_steer, self.max_steer)), speed


def make(
    line_csv, speed_scale: float, max_steer: float, v_max: float, dt: float, controller: str = "map", **params
) -> Spliner:
    """controller: "pp" (pure pursuit) or "map" (MAP with the identified model's steering lookup)."""
    import yaml

    from evaluation.common import REPO

    car = yaml.safe_load((REPO / "vehicle" / "params.yaml").read_text())
    if controller not in ("pp", "map"):
        raise ValueError(f"spliner controller must be pp or map, got {controller!r}")
    if controller == "map":
        params.setdefault("lut", SteerLookup(REPO / "evaluation" / "data" / "map_lut.csv"))
        params.setdefault("m_l1", 0.6)  # MAP's lab-track lookahead (module docstring)
    return Spliner(
        line=Line(line_csv),
        max_steer=max_steer,
        wheelbase=float(car["wheelbase_m"]),
        v_max=v_max,
        speed_scale=speed_scale,
        dt=dt,
        **params,
    )
