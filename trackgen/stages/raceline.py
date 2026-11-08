"""Minimum-curvature raceline, then wall widths along that line.

The track preparation, spline and minimum-curvature functions (import_track, prep_track,
interp_track, calc_splines, calc_spline_lengths, calc_head_curv_num, spline_approximation,
check_normals_crossing, side_of_line, dist_to_p, opt_min_curv) are adapted from TUM FTM's
trajectory_planning_helpers and global_racetrajectory_optimization
(https://github.com/TUMFTM), Copyright (C) Institute of Automotive Technology, Technical
University of Munich, licensed under the GNU LGPL v3.0. This file is distributed under the same
license; see LICENSES/LGPL-3.0.txt and LICENSES/GPL-3.0.txt.
"""
from __future__ import annotations

import math
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quadprog
from scipy import interpolate, optimize
from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter

from trackgen.paths import TrackPaths
from trackgen.plotting import draw_map, new_axes, show_plot


def _splev_xy(u: np.ndarray, tck: object, der: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 2D spline as a pair of float arrays."""
    raw: Any = splev(u, tck, der=der)
    return np.asarray(raw[0], dtype=float), np.asarray(raw[1], dtype=float)

def import_track(file_path: str,
                 imp_opts: dict,
                 width_veh: float) -> np.ndarray:
    """Load a centreline CSV as [x, y, width_right, width_left]."""
    csv_data_temp = np.loadtxt(file_path, comments='#', delimiter=',', skiprows=1)
    if np.shape(csv_data_temp)[1] == 3:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2] / 2
        w_tr_l = w_tr_r
    elif np.shape(csv_data_temp)[1] == 4:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2]
        w_tr_l = csv_data_temp[:, 3]
    elif np.shape(csv_data_temp)[1] == 10:
        print("INFO: Detected 10-column CSV. Using cols: 0(x), 1(y), 6(w_l), 7(w_r)")
        refline_ = csv_data_temp[:, 0:2]
        # 0.03 m safety margin subtracted from the stored widths.
        w_tr_l = csv_data_temp[:, 6]-0.03
        w_tr_r = csv_data_temp[:, 7]-0.03
    elif np.shape(csv_data_temp)[1] == 5:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 3]
        w_tr_l = csv_data_temp[:, 4]
    else:
        raise IOError(f"Track file cannot be read! Expected 3, 4, 5, or 10 columns, but got {np.shape(csv_data_temp)[1]}.")
    num_laps = imp_opts.get("num_laps", 1)
    refline_ = np.tile(refline_, (num_laps, 1))
    w_tr_r = np.tile(w_tr_r, num_laps)
    w_tr_l = np.tile(w_tr_l, num_laps)
    reftrack_imp = np.column_stack((refline_, w_tr_r, w_tr_l))
    if imp_opts.get("flip_imp_track", False): reftrack_imp = np.flipud(reftrack_imp)
    if imp_opts.get("set_new_start", False) and "new_start" in imp_opts:
        start_xy = imp_opts["new_start"]
        dists_sq = np.sum((reftrack_imp[:, :2] - start_xy)**2, axis=1)
        ind_start = np.argmin(dists_sq)
        reftrack_imp = np.roll(reftrack_imp, -ind_start, axis=0) # Use negative roll for correct shift
    w_tr_min = np.amin(reftrack_imp[:, 2] + reftrack_imp[:, 3])
    # Adjusted warning threshold slightly
    if w_tr_min < width_veh + 0.1: print(f"WARNING: Min track width {w_tr_min:.2f}m is close to vehicle width {width_veh:.2f}m")
    return reftrack_imp

def _raycast_widths(
    track: np.ndarray,
    normvec: np.ndarray,
    walls: tuple[np.ndarray, float, list[float]],
) -> None:
    """Overwrite track[:, 2:4] with free distances ray-cast along the normals.

    Normals point right, so +normal is w_r (column 2) and -normal is w_l (column 3).
    The inner wall is capped short of the centre of curvature so the Frenet factor
    1 - n*kappa stays positive across the whole width.
    """
    occupied, resolution, origin = walls
    closed = np.vstack((track[:, :2], track[0, :2]))
    el = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
    _psi, kappa = calc_head_curv_num(path=track[:, :2], el_lengths=el, is_closed=True, calc_curv=True)
    for i in range(len(track)):
        w_r, w_l = _wall_distances(track[i, :2], normvec[i], occupied, resolution, origin)
        safe_radius = 0.95 / (abs(float(kappa[i])) + 1e-6)
        if kappa[i] > 0.0:
            w_l = min(w_l, safe_radius)
        elif kappa[i] < 0.0:
            w_r = min(w_r, safe_radius)
        track[i, 2] = w_r
        track[i, 3] = w_l


def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict,
               debug: bool = True,
               min_width: float | None = None,
               enforce_smooth_closure: bool = True,  # accepted for call compatibility; not used here
               walls: tuple[np.ndarray, float, list[float]] | None = None) -> tuple:
    """Smooth the centreline, fit closed splines, and return normals.

    With `walls` = (occupied grid, resolution, origin) the widths are re-measured by
    ray-casting from the smoothed reference. The smoothing displaces the reference
    from the skeleton by up to the track half-width in hairpins, where the skeleton's
    symmetric nearest-wall widths no longer bound the free space around the new
    point; carrying them over lets the shift QP push the line through the inner wall.
    """
    reftrack_interp = spline_approximation(track=reftrack_imp,
                                           k_reg=reg_smooth_opts["k_reg"],
                                           s_reg=reg_smooth_opts["s_reg"],
                                           stepsize_prep=stepsize_opts["stepsize_prep"],
                                           stepsize_reg=stepsize_opts["stepsize_reg"],
                                           debug=debug)

    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = calc_splines(path=refpath_interp_cl, enforce_smoothness=True)

    if walls is not None:
        _raycast_widths(reftrack_interp, normvec_normalized_interp, walls)

    normals_crossing = check_normals_crossing(track=reftrack_interp,
                                              normvec_normalized=normvec_normalized_interp,
                                              horizon=10)
    if normals_crossing:
        print("WARNING: At least one pair of normals is crossed! This may lead to unexpected results.", file=sys.stderr)

    manipulated_track_width = False
    if min_width is not None:
        total_width = reftrack_interp[:, 2] + reftrack_interp[:, 3]
        inflate_indices = total_width < min_width
        if np.any(inflate_indices):
            manipulated_track_width = True
            diff = min_width - total_width[inflate_indices]
            reftrack_interp[inflate_indices, 2] += diff / 2.0
            reftrack_interp[inflate_indices, 3] += diff / 2.0

    if manipulated_track_width:
        print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation.",
              file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp

def normalize_psi(psi: np.ndarray | float) -> np.ndarray:
    """Wrap heading into [-pi, pi)."""
    is_array = isinstance(psi, np.ndarray)
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)
    if is_array:
        psi_out[psi_out >= math.pi] -= 2 * math.pi
        psi_out[psi_out < -math.pi] += 2 * math.pi
        return np.asarray(psi_out, dtype=float)
    if psi_out >= math.pi:
        psi_out -= 2 * math.pi
    elif psi_out < -math.pi:
        psi_out += 2 * math.pi
    return np.asarray(psi_out, dtype=float)

def create_raceline(refline: np.ndarray,
                    normvectors: np.ndarray,
                    alpha: np.ndarray,
                    stepsize_interp: float) -> tuple:
    """Apply lateral shifts and linearly resample the racing line."""
    # Calculate raceline points by applying alpha shifts
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    # Create a closed path
    raceline_cl = np.vstack((raceline, raceline[0]))

    el_lengths_raceline = np.sqrt(np.sum(np.power(np.diff(raceline_cl, axis=0), 2), axis=1))

    # Create cumulative distance vector (s) for this path
    s_raceline_nodes = np.cumsum(np.insert(el_lengths_raceline, 0, 0.0))
    total_dist_raceline = s_raceline_nodes[-1]

    if total_dist_raceline < 1e-3: # Path is essentially zero length
        print("Error: Raceline has near-zero length. Cannot interpolate.", file=sys.stderr)
        empty_path = np.zeros((0, 2)); empty_s = np.zeros(0); empty_len = np.zeros(0)
        return empty_path, empty_s, empty_len, empty_len

    # Create the new, high-resolution cumulative distance vector
    # Ensure num_interp_points is at least 2 for interpolation
    num_interp_points = max(2, math.ceil(total_dist_raceline / max(stepsize_interp, 1e-3)))
    s_raceline_interp_cl = np.linspace(0.0, total_dist_raceline, num_interp_points) # New s vector (closed)

    # Interpolate x and y coordinates
    # Use period=total_dist_raceline for circular interpolation, handles wrap-around
    x_interp = np.interp(s_raceline_interp_cl, s_raceline_nodes, raceline_cl[:, 0], period=total_dist_raceline)
    y_interp = np.interp(s_raceline_interp_cl, s_raceline_nodes, raceline_cl[:, 1], period=total_dist_raceline)

    # Return the unclosed path (remove the last point which duplicates the first)
    raceline_interp = np.column_stack((x_interp, y_interp))[:-1]
    s_raceline_interp = s_raceline_interp_cl[:-1]

    # Recalculate element lengths for the final *interpolated* path
    if len(raceline_interp) > 1:
        # Calculate lengths based on the interpolated points
        raceline_interp_cl_for_len = np.vstack((raceline_interp, raceline_interp[0]))
        el_lengths_final = np.sqrt(np.sum(np.power(np.diff(raceline_interp_cl_for_len, axis=0), 2), axis=1))
        el_lengths_raceline_interp_cl = el_lengths_final
    elif len(raceline_interp) == 1:
         el_lengths_raceline_interp_cl = np.array([0.0])
    else: # Handle empty case
         el_lengths_raceline_interp_cl = np.array([])
         s_raceline_interp = np.array([]) # Ensure s is also empty

    # spline_lengths_raceline (from original spline fit) is no longer relevant
    # Return the optimizer-grid element lengths instead
    spline_lengths_raceline = el_lengths_raceline # (N,) array

    return raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl

def side_of_line(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    """Return the side of point z relative to segment a->b."""
    # Ensure inputs are treated as arrays for indexing
    a = np.asarray(a); b = np.asarray(b); z = np.asarray(z)
    side = np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]))
    # Handle the case where the point is exactly on the line (result is 0)
    return float(side)

def interp_track(track: np.ndarray, stepsize: float) -> np.ndarray:
    """Linearly resample a closed track to a given spacing."""
    if len(track) < 2: return track # Cannot interpolate with less than 2 points
    track_cl = np.vstack((track, track[0]))
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    # Handle zero-length segments if any points are duplicated
    el_lengths_cl[el_lengths_cl < 1e-9] = 1e-9
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)
    total_dist = dists_cum_cl[-1]
    if total_dist < 1e-6: return track # Return original if total length is near zero
    no_points_interp_cl = math.ceil(total_dist / stepsize) + 1
    dists_interp_cl = np.linspace(0.0, total_dist, no_points_interp_cl)
    track_interp_cl = np.zeros((no_points_interp_cl, track_cl.shape[1]))
    for i in range(track_cl.shape[1]): # Interpolate all columns
        track_interp_cl[:, i] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, i])
    return track_interp_cl[:-1] # Return unclosed

def calc_spline_lengths(coeffs_x: np.ndarray, coeffs_y: np.ndarray, quickndirty: bool = False, no_interp_points: int = 15) -> np.ndarray:
    """Estimate each cubic-spline segment length."""
    if coeffs_x.shape[0] != coeffs_y.shape[0]: raise RuntimeError("Coefficient matrices must have the same length!")
    if coeffs_x.ndim == 1: coeffs_x = np.expand_dims(coeffs_x, 0)
    if coeffs_y.ndim == 1: coeffs_y = np.expand_dims(coeffs_y, 0)
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)
    if quickndirty:
        for i in range(no_splines): spline_lengths[i] = math.sqrt(math.pow(np.sum(coeffs_x[i]) - coeffs_x[i, 0], 2) + math.pow(np.sum(coeffs_y[i]) - coeffs_y[i, 0], 2))
    else:
        t_steps = np.linspace(0.0, 1.0, no_interp_points)
        spl_coords = np.zeros((no_interp_points, 2))
        t_pow = np.vstack([t_steps**0, t_steps**1, t_steps**2, t_steps**3]).T # T^0, T^1, T^2, T^3 columns
        for i in range(no_splines):
            spl_coords[:, 0] = t_pow @ coeffs_x[i, :] # Matrix multiplication for efficiency
            spl_coords[:, 1] = t_pow @ coeffs_y[i, :]
            diffs = np.diff(spl_coords, axis=0)
            seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            spline_lengths[i] = np.sum(seg_lengths)
    return spline_lengths

def calc_splines(path: np.ndarray, el_lengths: np.ndarray | None = None, psi_s: float | None = None, psi_e: float | None = None, use_dist_scaling: bool = True, enforce_smoothness: bool = True) -> tuple: #type: ignore
    """Fit closed cubic splines and unit normals to a path."""
    if len(path) < 2: raise ValueError("Path must have at least 2 points for spline calculation.")
    if np.all(np.isclose(path[0], path[-1])) and psi_s is None: closed = True
    else: closed = False
    if not closed and (psi_s is None or psi_e is None): raise RuntimeError("Headings must be provided for unclosed spline calculation!")
    if el_lengths is not None and path.shape[0] != el_lengths.size + 1: raise RuntimeError("el_lengths input must be one element smaller than path input!")
    if el_lengths is None:
        lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    else:
        lengths = np.copy(el_lengths)
    if use_dist_scaling and closed:
        lengths = np.append(lengths, lengths[0])
    no_splines = path.shape[0] - 1
    if use_dist_scaling:
        lengths[lengths < 1e-9] = 1e-9
        scaling = lengths[:-1] / lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)
    el_lengths = lengths
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))
    template_M = np.array([[1,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0],[0,1,2,3,0,-1,0,0],[0,0,2,6,0,0,-2,0]])
    for i in range(no_splines):
        j = i * 4
        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M
            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)
        else: M[j: j + 2, j: j + 4] = [[1,0,0,0],[1,1,1,1]]
        b_x[j: j + 2] = [[path[i,0]], [path[i + 1,0]]]
        b_y[j: j + 2] = [[path[i,1]], [path[i + 1,1]]]
    if not closed:
        assert psi_s is not None and psi_e is not None
        M[-2, 1] = 1
        el_s = 1.0 if not use_dist_scaling else float(lengths[0])
        b_x[-2] = math.cos(psi_s + math.pi / 2) * el_s
        b_y[-2] = math.sin(psi_s + math.pi / 2) * el_s
        M[-1, -4:] = [0,1,2,3]
        el_e = 1.0 if not use_dist_scaling else float(lengths[-1])
        b_x[-1] = math.cos(psi_e + math.pi / 2) * el_e
        b_y[-1] = math.sin(psi_e + math.pi / 2) * el_e
    elif enforce_smoothness: # Only apply smoothness constraints if requested
        if no_splines > 0: # Ensure scaling[-1] exists
            M[-2, 1] = scaling[-1]
            M[-2, -3:] = [-1, -2, -3]
            M[-1, 2] = 2 * math.pow(scaling[-1], 2)
            M[-1, -2:] = [-2, -6]
        else: # Handle case of single point path? May not be meaningful.
             pass
    # Solve system
    try: x_les = np.squeeze(np.linalg.solve(M, b_x)); y_les = np.squeeze(np.linalg.solve(M, b_y))
    except np.linalg.LinAlgError as e:
         print(f"Error: Could not solve spline equations: {e}");
         print("Matrix M might be singular. Check input points (duplicates, collinearity?).")
         raise
    coeffs_x = np.reshape(x_les, (no_splines, 4)); coeffs_y = np.reshape(y_les, (no_splines, 4))
    # Calculate normals, handle potential zero vectors
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1) # Tangent vector is (coeffs_x[:,1], coeffs_y[:,1])
    norm = np.linalg.norm(normvec, axis=1)
    # Initialize normvec_normalized with zeros or a default direction
    normvec_normalized = np.zeros_like(normvec)
    # Calculate normalized vectors only where norm is sufficiently large
    valid_norm_indices = norm > 1e-9
    normvec_normalized[valid_norm_indices] = normvec[valid_norm_indices] / norm[valid_norm_indices, np.newaxis]
    # Handle zero-norm cases (e.g., if a segment has zero length or direction is undefined)
    # Such segments keep a zero normal; warn so the caller can inspect the input.
    if np.any(~valid_norm_indices):
        print(f"Warning: {np.sum(~valid_norm_indices)} zero-length or undefined tangent segments found in calc_splines. Normals set to [0,0].")

    return coeffs_x, coeffs_y, M, normvec_normalized

def check_normals_crossing(track: np.ndarray, normvec_normalized: np.ndarray, horizon: int = 10) -> bool:
    """Return True if neighbouring track normals cross."""
    no_points = track.shape[0]
    if no_points < 3: return False # Need at least 3 points to check crossing
    if horizon >= no_points / 2: horizon = int(no_points / 2) - 1 # Adjust horizon
    if horizon < 1 : return False # Cannot check if horizon is too small
    les_mat = np.zeros((2, 2))
    idx_list = list(range(0, no_points))
    idx_list = idx_list[-horizon:] + idx_list + idx_list[:horizon]
    for idx in range(no_points):
        idx_neighbours = idx_list[idx:idx + 2 * horizon + 1]; del idx_neighbours[horizon]
        idx_neighbours = np.array(idx_neighbours)
        # Ensure normvec_normalized[idx] is valid before cross product
        if np.linalg.norm(normvec_normalized[idx]) < 1e-9: continue

        cross_prods = np.cross(normvec_normalized[idx], normvec_normalized[idx_neighbours])
        is_collinear_b = np.isclose(cross_prods, 0.0, atol=1e-6) # Add tolerance
        idx_neighbours_rel = idx_neighbours[~is_collinear_b] # Use boolean indexing
        for idx_comp in idx_neighbours_rel:
            const = track[idx_comp, :2] - track[idx, :2]
            les_mat[:, 0] = normvec_normalized[idx]; les_mat[:, 1] = -normvec_normalized[idx_comp]
            try: lambdas = np.linalg.solve(les_mat, const)
            except np.linalg.LinAlgError: continue
            # Check if intersection is within track width segment
            w_r_idx = track[idx, 2]; w_l_idx = track[idx, 3]
            w_r_comp = track[idx_comp, 2]; w_l_comp = track[idx_comp, 3]
            # Use small tolerance epsilon
            eps = 1e-6
            # Check if intersection point lies *between* the centerline points along the normal
            # Lambda represents distance along normal vector from centerline
            if (-w_l_idx - eps <= lambdas[0] <= w_r_idx + eps and
                -w_l_comp - eps <= lambdas[1] <= w_r_comp + eps):
                # Additional check: ensure intersection isn't just the start/end points being close
                if abs(lambdas[0]) > 1e-3 and abs(lambdas[1]) > 1e-3: # Heuristic threshold
                     return True
    return False

def dist_to_p(t_glob: float, path_coeffs: tuple, p: np.ndarray) -> float:
    """Distance from p to the spline point at parameter t."""
    # Ensure t_glob is within [0, 1] for splev
    t_eval = np.clip(t_glob, 0.0, 1.0)
    try:
        s_raw = interpolate.splev(t_eval, path_coeffs)
        s = np.array(s_raw).flatten()
        return float(np.linalg.norm(p - s))
    except ValueError as e:
        print(f"Warning: splev failed with t={t_eval}. Error: {e}")
        return float("inf")

def spline_approximation(track: np.ndarray, k_reg: int = 3, s_reg: int = 10, stepsize_prep: float = 1.0, stepsize_reg: float = 3.0, debug: bool = False) -> np.ndarray:
    """Fit a smooth closed spline and interpolate track widths onto it."""
    track_interp = interp_track(track=track, stepsize=stepsize_prep)
    if len(track_interp) < k_reg + 1: # Ensure enough points for splprep
         print(f"Warning: Upsampling track for splprep (need {k_reg+1}, have {len(track_interp)})")
         num_needed = k_reg + 1
         track_cl_orig = np.vstack((track, track[0]))
         el_len_orig = np.sqrt(np.sum(np.power(np.diff(track_cl_orig[:, :2], axis=0), 2), axis=1))
         dist_cum_orig = np.cumsum(np.insert(el_len_orig,0,0.0))
         dist_fine = np.linspace(0, dist_cum_orig[-1], num_needed)
         x_fine = np.interp(dist_fine, dist_cum_orig, track_cl_orig[:,0])
         y_fine = np.interp(dist_fine, dist_cum_orig, track_cl_orig[:,1])
         track_interp_cl = np.column_stack([x_fine, y_fine])
    else:
         track_interp_cl = np.vstack((track_interp[:,:2], track_interp[0,:2])) # Use only xy for splprep

    track_cl = np.vstack((track, track[0]))
    no_points_track_cl = track_cl.shape[0]
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    el_lengths_cl[el_lengths_cl < 1e-9] = 1e-9 # Avoid zero lengths
    dists_cum_cl = np.cumsum(el_lengths_cl); dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)
    total_len_orig = dists_cum_cl[-1]
    if total_len_orig < 1e-6: return track # Return original if too short

    # Ensure u parameterization is roughly uniform if points are unevenly spaced
    u_param = np.linspace(0, 1, len(track_interp_cl)) # Use uniform if unsure
    try:
        tck_cl, u_eval = interpolate.splprep([track_interp_cl[:, 0], track_interp_cl[:, 1]], u=u_param, k=k_reg, s=s_reg, per=1)
    except Exception as e:
        print(f"Error during splprep: {e}. Check input points for duplicates or NaNs.")
        raise

    # Estimate length more accurately from the resulting spline
    no_points_lencalc_cl = max(100, math.ceil(total_len_orig) * 4) # Use more points for length calc
    path_smoothed_lencalc = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_lencalc_cl), tck_cl)).T
    len_path_smoothed_est = np.sum(np.sqrt(np.sum(np.power(np.diff(path_smoothed_lencalc, axis=0), 2), axis=1)))

    no_points_reg_cl = math.ceil(len_path_smoothed_est / stepsize_reg) + 1
    t_reg = np.linspace(0.0, 1.0, no_points_reg_cl)
    path_smoothed = np.array(interpolate.splev(t_reg, tck_cl)).T[:-1] # Evaluate at new points, remove duplicate end

    # Find closest points on smoothed spline to original points
    dists_cl = np.zeros(no_points_track_cl)
    closest_point_cl = np.zeros((no_points_track_cl, 2)); closest_t_glob_cl = np.zeros(no_points_track_cl)
    t_glob_guess_cl = dists_cum_cl / total_len_orig # Use original distances for guess

    for i in range(no_points_track_cl):
         # Use optimize.minimize for potentially better results than fmin
         res = optimize.minimize(dist_to_p, x0=t_glob_guess_cl[i], args=(tck_cl, track_cl[i, :2]), method='Nelder-Mead', bounds=[(0,1)])
         if res.success:
             closest_t_glob_cl[i] = res.x[0]
             dists_cl[i] = res.fun
         else: # Fallback or error handling
             print(f"Warning: Closest point optimization failed for index {i}. Using guess.")
             closest_t_glob_cl[i] = t_glob_guess_cl[i]
             dists_cl[i] = dist_to_p(closest_t_glob_cl[i], tck_cl, track_cl[i, :2])
         closest_point_cl[i] = interpolate.splev(closest_t_glob_cl[i], tck_cl)

    if debug: print(f"Spline approximation: mean dev {np.mean(dists_cl):.2f}m, max dev {np.amax(np.abs(dists_cl)):.2f}m")

    # Interpolate widths based on closest points
    sides = np.zeros(no_points_track_cl - 1)
    for i in range(no_points_track_cl - 1): sides[i] = side_of_line(a=track_cl[i, :2], b=track_cl[i+1, :2], z=closest_point_cl[i])
    sides_cl = np.hstack((sides, sides[0]))
    w_tr_right_new_cl = track_cl[:, 2] + sides_cl * dists_cl
    w_tr_left_new_cl = track_cl[:, 3] - sides_cl * dists_cl
    # Ensure t_glob values are monotonic for interpolation
    closest_t_glob_cl = np.mod(np.unwrap(closest_t_glob_cl * 2 * np.pi) / (2*np.pi), 1.0) # Unwrap angles, normalize to [0,1)
    sort_indices = np.argsort(closest_t_glob_cl)
    closest_t_glob_cl_sorted = closest_t_glob_cl[sort_indices]
    # Handle non-unique t values by averaging widths at duplicate t's
    unique_t, unique_indices = np.unique(closest_t_glob_cl_sorted, return_index=True)
    w_tr_right_unique = np.array([np.mean(w_tr_right_new_cl[sort_indices][closest_t_glob_cl_sorted == t]) for t in unique_t])
    w_tr_left_unique = np.array([np.mean(w_tr_left_new_cl[sort_indices][closest_t_glob_cl_sorted == t]) for t in unique_t])

    # Interpolate using unique, sorted t values
    w_tr_right_smoothed_cl = np.interp(t_reg, unique_t, w_tr_right_unique, period=1.0) # Use period=1.0 for circular interp
    w_tr_left_smoothed_cl = np.interp(t_reg, unique_t, w_tr_left_unique, period=1.0)

    track_reg = np.column_stack((path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]))
    if track_cl.shape[1] >= 5: # Handle banking
        banking_new_cl = track_cl[:, 4][sort_indices]
        banking_unique = np.array([np.mean(banking_new_cl[closest_t_glob_cl_sorted == t]) for t in unique_t])
        banking_smoothed_cl = np.interp(t_reg, unique_t, banking_unique, period=1.0)
        track_reg = np.column_stack((track_reg, banking_smoothed_cl[:-1]))
    return track_reg

def calc_head_curv_num(path: np.ndarray, el_lengths: np.ndarray, is_closed: bool, stepsize_psi_preview: float = 1.0, stepsize_psi_review: float = 1.0, stepsize_curv_preview: float = 2.0, stepsize_curv_review: float = 2.0, calc_curv: bool = True) -> tuple:
    """Numerically estimate heading and curvature along a path."""
    # (Input checks omitted)
    no_points = path.shape[0]
    psi = np.zeros(no_points)
    kappa = np.zeros(no_points)

    if is_closed:
        if len(el_lengths)==0: return psi, kappa # Handle empty case
        avg_step = np.mean(el_lengths)
        if avg_step < 1e-6: avg_step = 1.0 # Avoid division by zero
        ind_prev_psi = max(1, round(stepsize_psi_preview / avg_step))
        ind_rev_psi = max(1, round(stepsize_psi_review / avg_step))
        ind_prev_curv = max(1, round(stepsize_curv_preview / avg_step))
        ind_rev_curv = max(1, round(stepsize_curv_review / avg_step))

        # Ensure padding indices don't exceed array bounds
        ind_rev_psi = min(ind_rev_psi, no_points -1)
        ind_prev_psi = min(ind_prev_psi, no_points -1)
        ind_rev_curv = min(ind_rev_curv, no_points -1)
        ind_prev_curv = min(ind_prev_curv, no_points -1)

        path_padded = np.vstack((path[-ind_rev_psi:], path, path[:ind_prev_psi]))
        dx = path_padded[ind_rev_psi+ind_prev_psi:, 0] - path_padded[:-ind_rev_psi-ind_prev_psi, 0]
        dy = path_padded[ind_rev_psi+ind_prev_psi:, 1] - path_padded[:-ind_rev_psi-ind_prev_psi, 1]
        psi = np.arctan2(dy, dx) # Assuming 0 is EAST # MODIFIED
        psi = normalize_psi(psi)

        if calc_curv:
            psi_padded = np.concatenate((psi[-ind_rev_curv:], psi, psi[:ind_prev_curv]))
            delta_psi = normalize_psi(psi_padded[ind_rev_curv+ind_prev_curv:] - psi_padded[:-ind_rev_curv-ind_prev_curv])

            s_cum = np.cumsum(el_lengths); s_cum = np.insert(s_cum, 0, 0.0)
            if len(s_cum) < 2: return psi, kappa # Cannot calculate s_rev if only one element
            s_rev = s_cum[-1] - s_cum[::-1][1:] # Cumulative dist backwards from end point
            s_rev = np.insert(s_rev, len(s_rev), 0.0) # Add 0 at the end
            s_padded = np.concatenate((s_rev[-ind_rev_curv-1:-1], s_cum[:-1], s_cum[-1] + s_cum[1:ind_prev_curv+1]))


            delta_s = s_padded[ind_rev_curv+ind_prev_curv:] - s_padded[:-ind_rev_curv-ind_prev_curv]
            delta_s[np.abs(delta_s) < 1e-6] = 1e-6 # Avoid division by zero
            kappa = delta_psi / delta_s
    else: # Unclosed path
        if no_points < 2: return psi, kappa
        dx = np.diff(path[:, 0]); dy = np.diff(path[:, 1])
        psi_segments = np.arctan2(dy, dx) - math.pi / 2
        psi[0] = psi_segments[0]
        if no_points > 2:
             psi[-1] = psi_segments[-1]
             psi[1:-1] = (psi_segments[:-1] + psi_segments[1:]) / 2.0 # Average angles
        else: # Only 2 points
             psi[1] = psi_segments[0]
        psi = normalize_psi(psi)
        if calc_curv:
            delta_psi = normalize_psi(np.diff(psi))
            safe_el = el_lengths.copy()
            safe_el[safe_el < 1e-6] = 1e-6
            if no_points > 2:
                 avg_el_lengths = (safe_el[:-1] + safe_el[1:]) / 2.0
                 kappa[1:-1] = delta_psi[1:] / avg_el_lengths # Central difference approx
                 kappa[0] = delta_psi[0] / safe_el[0]     # Forward difference
                 kappa[-1] = delta_psi[-1] / safe_el[-1]   # Backward difference
            else: # Only 2 points
                 kappa[0] = delta_psi[0] / safe_el[0]
                 kappa[1] = kappa[0] # Assume constant curvature

    return psi, kappa


def opt_min_curv(reftrack: np.ndarray,
                 normvectors: np.ndarray,
                 A: np.ndarray,
                 kappa_bound: float,
                 w_veh: float,
                 print_debug: bool = False,
                 plot_debug: bool = False,
                 closed: bool = True,
                 psi_s: float | None = None,
                 psi_e: float | None = None,
                 fix_s: bool = False,
                 fix_e: bool = False) -> tuple:
    """Solve the minimum-curvature lateral-shift problem."""
    no_points = reftrack.shape[0]

    no_splines = no_points
    if not closed:
        no_splines -= 1

    # check inputs
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")

    if (no_points * 4 != A.shape[0] and closed) or (no_splines * 4 != A.shape[0] and not closed)\
            or A.shape[0] != A.shape[1]:
        raise RuntimeError("Spline equation system matrix A has wrong dimensions!")

    # create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
    # information
    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])

    # create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
    # information
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])

    # invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    # set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]  # close spline

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]

    # set up q_x and q_y matrices including the point coordinate information
    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]

    # for unclosed tracks, specify start- and end-heading constraints
    if not closed:
        assert psi_s is not None and psi_e is not None
        q_x[-2, 0] = math.cos(psi_s + math.pi / 2)
        q_y[-2, 0] = math.sin(psi_s + math.pi / 2)

        q_x[-1, 0] = math.cos(psi_e + math.pi / 2)
        q_y[-1, 0] = math.sin(psi_e + math.pi / 2)

    # set up P_xx, P_xy, P_yy matrices
    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calculate curvature denominator
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)                          # divide where not zero (diag elements)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2   # make H symmetric

    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = f_x + f_xy + f_y
    f = np.squeeze(f)   # remove non-singleton dimensions

    # ------------------------------------------------------------------------------------------------------------------
    # KAPPA CONSTRAINTS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)

    # this part is multiplied by alpha within the optimization (variable part)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

    # original curvature part (static part)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
    con_stack = np.append(con_ge, con_le)

    # ------------------------------------------------------------------------------------------------------------------
    # CALL QUADRATIC PROGRAMMING ALGORITHM -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """
    quadprog interface description taken from
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py

    Solve a Quadratic Program defined as:

        minimize
            (1/2) * alpha.T * H * alpha + f.T * alpha

        subject to
            G * alpha <= h
            A * alpha == b

    using quadprog <https://pypi.python.org/pypi/quadprog/>.

    Parameters
    ----------
    H : numpy.array
        Symmetric quadratic-cost matrix.
    f : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).

    Returns
    -------
    alpha : numpy.array
            Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    The quadprog solver only considers the lower entries of `H`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """

    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # constrain resulting path to reference line at start- and end-point for open tracks
    if not closed and fix_s:
        dev_max_left[0] = 0.05
        dev_max_right[0] = 0.05

    if not closed and fix_e:
        dev_max_left[-1] = 0.05
        dev_max_right[-1] = 0.05

    # check that there is space remaining between left and right maximum deviation (both can be negative as well!)
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problem not solvable, track might be too small to run with current safety distance!")

    # consider value boundaries (-dev_max_left <= alpha <= dev_max_right)
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # save start time
    t_start = time.perf_counter()

    # solve problem (quadprog) -----------------------------------------------------------------------------------------
    alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    # print runtime into console window
    if print_debug:
        print("Solver runtime opt_min_curv: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE CURVATURE ERROR ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate curvature once based on original linearization and once based on a new linearization around the solution
    q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
    q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

    x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
    y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

    x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
    y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

    curv_orig_lin = np.zeros(no_points)
    curv_sol_lin = np.zeros(no_points)

    for i in range(no_points):
        curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                          / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
        curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                           / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)

    if plot_debug:
        plt.plot(curv_orig_lin)
        plt.plot(curv_sol_lin)
        plt.legend(("original linearization", "solution based linearization"))
        plt.show()

    # calculate maximum curvature error
    curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

    return alpha_mincurv, curv_error_max

def compute_path_length(path_xy: np.ndarray) -> float:
    """Return the length of a closed polyline."""
    if len(path_xy) < 2:
        return 0.0
    path_closed = np.vstack([path_xy, path_xy[0]])
    distances = np.sqrt(np.sum(np.power(np.diff(path_closed, axis=0), 2), axis=1))
    return float(np.sum(distances))


STEPSIZE_OPTS = {
    "stepsize_prep": 0.1,
    "stepsize_reg": 0.05,
    "stepsize_interp_after_opt": 0.005,
}
REG_SMOOTH_OPTS = {"k_reg": 3, "s_reg": 5.0}
IMP_OPTS: dict[str, Any] = {
    "flip_imp_track": False,
    "set_new_start": False,
    "new_start": [0.0, -47.0],
    "min_track_width": None,
    "num_laps": 1,
}
VEH_WIDTH = 0.29
V_MAX = 4.0
A_MAX_LAT = 2.5
WIDTH_OPT_MINCURV = 0.9
KAPPA_BOUND = 0.8
MINCURV_STEP_M = 0.15
MINCURV_MAX_ITERS = 40
MINCURV_TOL_M = 0.1
MINCURV_GRID_M = 0.25
RAYCAST_RESOLUTION = 0.05
RAYCAST_SMOOTHING = 0.1
WALL_STEP_M = 0.01
WALL_MAX_M = 3.0


def _wall_distances(
    point: np.ndarray,
    normal: np.ndarray,
    occupied: np.ndarray,
    resolution: float,
    origin: list[float],
) -> tuple[float, float]:
    """Conservative free distance left and right of a path point along its normal.

    Walks outward in WALL_STEP_M steps until an occupied or out-of-map cell and
    subtracts half a cell, so the result never over-reports the mapped free space.
    """
    height, width = occupied.shape

    def blocked(x: float, y: float) -> bool:
        col = int((x - origin[0]) / resolution)
        row = height - 1 - int((y - origin[1]) / resolution)
        return row < 0 or row >= height or col < 0 or col >= width or bool(occupied[row, col])

    def free_distance(direction: np.ndarray) -> float:
        d = 0.0
        while d < WALL_MAX_M and not blocked(point[0] + d * direction[0], point[1] + d * direction[1]):
            d += WALL_STEP_M
        return max(0.0, d - 0.5 * resolution)

    return free_distance(normal), free_distance(-normal)


def _solve_shift(
    paths: TrackPaths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the minimum-curvature line by trust-region steps from the smoothed centreline.

    Returns the imported centreline, the final reference (0.25 m grid, ray-cast
    widths), its normals, and the lateral shifts to apply to that reference.
    """
    width_opt = WIDTH_OPT_MINCURV
    reftrack_imp = import_track(str(paths.centreline_csv), IMP_OPTS, VEH_WIDTH)
    _image, resolution, origin = paths.load_occupancy()
    # Iterate on a coarse grid: the QP is dense in the point count, and the final
    # line is resampled at STEPSIZE_OPTS resolution by raycast_table anyway.
    iter_steps = {**STEPSIZE_OPTS, "stepsize_reg": MINCURV_GRID_M}
    reftrack_interp, normvec, a_interp, _cx, _cy = prep_track(
        reftrack_imp=reftrack_imp,
        reg_smooth_opts=dict(REG_SMOOTH_OPTS),
        stepsize_opts=iter_steps,
        debug=False,
        min_width=IMP_OPTS.get("min_track_width"),
        enforce_smooth_closure=True,
        walls=(paths.occupied_grid(), resolution, origin),
    )
    if a_interp is None:
        raise RuntimeError("spline matrix A missing for minimum-curvature")
    # The QP linearises curvature about the reference, so one solve is only valid
    # for shifts small against the local radius. A hairpin reference can have a
    # local radius below the shift the racing line needs (it hugs the outer wall
    # and turns in under a metre), and a single full shift folds it. Take
    # trust-region steps instead: shift at most MINCURV_STEP_M, re-smooth, re-fit
    # splines, re-measure the walls, and repeat until the remaining shift is small.
    walls = (paths.occupied_grid(), resolution, origin)
    ref = reftrack_interp
    for iteration in range(MINCURV_MAX_ITERS):
        alpha = np.asarray(
            opt_min_curv(
                reftrack=ref,
                normvectors=normvec,
                A=a_interp,
                kappa_bound=KAPPA_BOUND,
                w_veh=width_opt,
                print_debug=False,
                plot_debug=False,
                closed=True,
            )[0]
        ).flatten()
        shift = float(np.max(np.abs(alpha)))
        print(f"  min-curvature step {iteration + 1}: max shift {shift:.3f} m", flush=True)
        if shift <= MINCURV_TOL_M or iteration == MINCURV_MAX_ITERS - 1:
            # The returned shift applies to this reference; keep it inside the
            # trust region so the final line cannot fold either.
            alpha = np.clip(alpha, -MINCURV_STEP_M, MINCURV_STEP_M)
            break
        step = np.clip(alpha, -MINCURV_STEP_M, MINCURV_STEP_M)
        moved = np.column_stack(
            (ref[:, :2] + step[:, None] * normvec, ref[:, 2] - step, ref[:, 3] + step)
        )
        ref = interp_track(moved, MINCURV_GRID_M)
        _cx, _cy, a_interp, normvec = calc_splines(
            path=np.vstack((ref[:, :2], ref[0, :2])), enforce_smoothness=True
        )
        _raycast_widths(ref, normvec, walls)
    return reftrack_imp, ref, normvec, alpha


def lap_time(raceline_xy: np.ndarray, el_lengths: np.ndarray) -> float:
    """Return the curvature-limited lap-time estimate used as opt_time."""
    _, kappa = calc_head_curv_num(
        path=raceline_xy, el_lengths=el_lengths, is_closed=True, calc_curv=True
    )
    if len(kappa) != len(raceline_xy):
        kappa = np.zeros(len(raceline_xy))
    vx = np.zeros_like(kappa)
    for i in range(len(vx)):
        vx[i] = min(V_MAX, math.sqrt(A_MAX_LAT / (abs(kappa[i]) + 1e-6)))
    safe_vx = np.maximum(vx, 1e-3)
    if len(el_lengths) == len(safe_vx):
        return float(np.sum(el_lengths / safe_vx))
    total = compute_path_length(raceline_xy)
    avg_v = float(np.mean(safe_vx)) if len(safe_vx) else V_MAX
    return total / avg_v if avg_v > 1e-6 else 0.0


def raycast_table(
    xy: np.ndarray,
    occupied: np.ndarray,
    resolution: float,
    origin: list[float],
) -> pd.DataFrame:
    """Fit a closed spline to xy, resample it, and measure the wall clearance at every station."""
    x_raw = xy[:, 0].copy()
    y_raw = xy[:, 1].copy()
    # A periodic fit needs the last point to equal the first exactly.
    if np.hypot(x_raw[0] - x_raw[-1], y_raw[0] - y_raw[-1]) < RAYCAST_RESOLUTION:
        x_raw, y_raw = x_raw[:-1], y_raw[:-1]
    x_raw = np.append(x_raw, x_raw[0])
    y_raw = np.append(y_raw, y_raw[0])
    tck, _u = splprep([x_raw, y_raw], s=RAYCAST_SMOOTHING, per=True, k=3)
    u_test = np.linspace(0, 1, len(x_raw) * 10)
    x_t, y_t = _splev_xy(u_test, tck)
    total_len = float(np.sum(np.hypot(np.diff(x_t), np.diff(y_t))))
    num_points = int(total_len / RAYCAST_RESOLUTION)
    u_new = np.linspace(0, 1, num_points)[:-1]
    x_new, y_new = _splev_xy(u_new, tck)
    s_new = np.concatenate((np.array([0.0]), np.cumsum(np.hypot(np.diff(x_new), np.diff(y_new)))))
    dx, dy = _splev_xy(u_new, tck, der=1)
    d2x, d2y = _splev_xy(u_new, tck, der=2)
    track_yaw = np.arctan2(dy, dx)
    kappa = (dx * d2y - dy * d2x) / ((dx ** 2 + dy ** 2) ** 1.5)
    kappa = np.asarray(savgol_filter(kappa, window_length=41, polyorder=3), dtype=float)
    n_x, n_y = -dy, dx
    norm = np.hypot(n_x, n_y)
    n_x = n_x / norm
    n_y = n_y / norm
    w_l_raw = np.zeros(len(x_new))
    w_r_raw = np.zeros(len(x_new))
    for i in range(len(x_new)):
        w_l_raw[i], w_r_raw[i] = _wall_distances(
            np.array([x_new[i], y_new[i]]),
            np.array([n_x[i], n_y[i]]),
            occupied,
            resolution,
            origin,
        )
    # The Frenet factor 1 - n * kappa vanishes only toward the centre of
    # curvature, so cap the inner width at 0.95 of the radius; the outer wall
    # stays where the walk found it.
    safe_radius = 0.95 / (np.abs(kappa) + 1e-6)
    w_l_safe = np.where(kappa > 0.0, np.minimum(w_l_raw, safe_radius), w_l_raw)
    w_r_safe = np.where(kappa < 0.0, np.minimum(w_r_raw, safe_radius), w_r_raw)
    df = pd.DataFrame(
        {
            "x": x_new,
            "y": y_new,
            "s": s_new,
            "s_norm": s_new / s_new[-1],
            "track_yaw_rad": track_yaw,
            "kappa": kappa,
            "w_l": w_l_safe,
            "w_r": w_r_safe,
            "n_x": n_x,
            "n_y": n_y,
        }
    )
    return df


def draw_walls(ax: Any, df: pd.DataFrame, label: str) -> None:
    """Draw the wall clearances of a 10-column line table on ax."""
    ax.plot(df["x"] + df["n_x"] * df["w_l"], df["y"] + df["n_y"] * df["w_l"], "g-", lw=0.8, label=label)
    ax.plot(df["x"] - df["n_x"] * df["w_r"], df["y"] - df["n_y"] * df["w_r"], "g-", lw=0.8)


def run(paths: TrackPaths, direction: str, plot: bool) -> None:
    """Write raceline.csv and print a curvature-limited lap-time estimate."""
    cl_csv = paths.centreline_csv
    if not cl_csv.exists():
        raise FileNotFoundError(f"missing centreline: {cl_csv}")
    image, resolution, origin = paths.load_occupancy()
    print("solving minimum curvature")
    ref_imp, ref_mc, norm_mc, alpha_mc = _solve_shift(paths)
    raceline_xy, s_rl, _sl, el_cl = create_raceline(
        refline=ref_mc[:, :2],
        normvectors=norm_mc,
        alpha=alpha_mc,
        stepsize_interp=STEPSIZE_OPTS["stepsize_interp_after_opt"],
    )
    t_final = lap_time(raceline_xy, el_cl)
    mc_len = compute_path_length(raceline_xy)
    print(f"Raceline Length: {mc_len:.2f} m")
    if plot:
        fig, ax = new_axes((12, 12))
        draw_map(ax, image, resolution, origin)
        ax.plot(ref_imp[:, 0], ref_imp[:, 1], color="gold", ls="--", lw=1.2, label="centreline")
        ax.plot(raceline_xy[:, 0], raceline_xy[:, 1], "b-", lw=1.5, label="minimum curvature")
        ax.scatter([raceline_xy[0, 0]], [raceline_xy[0, 1]], c="lime", s=80, zorder=5, label="s = 0")
        ax.set_title(f"minimum curvature ({direction}, curvature-limited time {t_final:.2f}s)")
        ax.legend(loc="upper right")
        show_plot(fig, True)

    table = raycast_table(raceline_xy, paths.occupied_grid(), resolution, origin)
    table.to_csv(paths.raceline_csv, index=False, float_format="%.4f")
    print(f"wrote {paths.raceline_csv}")
    if plot:
        fig, ax = new_axes((12, 12))
        draw_map(ax, image, resolution, origin)
        ax.plot(table["x"], table["y"], "b-", lw=1.5, label="raceline")
        draw_walls(ax, table, "wall clearances")
        ax.scatter([table["x"][0]], [table["y"][0]], c="lime", s=80, zorder=5, label="s = 0")
        ax.set_title("raceline with wall widths")
        ax.legend(loc="upper right")
        show_plot(fig, True)

