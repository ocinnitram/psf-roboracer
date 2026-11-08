# sim

First-party C plant and env (`ocean/roboracer/roboracer.h`) behind a ctypes
`rr_*` ABI. The plant is a Pacejka single-track chassis with actuator lags,
integrated at 1 ms with RK4. Each env ray-marches LiDAR on a signed Euclidean
distance field, tests collision against that field plus analytic rectangles, and
scores progress on the raceline. Python loads `sim/build/libroboracer.so` through
`sim/python/roboracer_sim.py`. Every car quantity arrives as a `my_init` kwarg
built from `vehicle/params.yaml`; tracks are the blobs written by `tools/compile_track.py`.

```
ocean/roboracer/   roboracer.h (plant + env), renderer.c (display)
                    binding.c + shim.c (rr_* ABI)
python/            roboracer_sim.py: VecEnv, Renderer, car_kwargs
tools/             compile_track.py: sf/tracks/<map>/ -> <map>.blob
tests/             C tests (dynamics, env, EDT) and the no-op raylib stub
externals/         core-math: correctly rounded tanf, atanf, asinf, atan2f (vendored)
Makefile           make | make RENDER=1 | make test | make clean
```

The plant rounds the same on every machine: it builds with `-ffp-contract=off`
and calls CORE-MATH's `cr_tanf`, `cr_atanf`, `cr_asinf` and `cr_atan2f` instead
of libm's, which glibc changed between releases (2.35 and 2.44 differ in the last
bit, and one bit in the tyre force changes the layout's outcome). Together with
the filter's pins (`sf/README.md`) an evaluation reproduces on another machine.

## Build

```bash
make               # sim/build/libroboracer.so against tests/raylib.h (render is a no-op)
make RENDER=1      # fetch raylib 5.5 into sim/.deps/raylib (gitignored) and link it
make test          # build and run the three C tests
make clean
```

`rr_render_available()` is 1 only for the `RENDER=1` build.

Compile a track (blob is gitignored):

```bash
uv run --frozen python sim/tools/compile_track.py lab-20260906
uv run --frozen python sim/tools/compile_track.py lab-20260906 --no-plot
```

## `rr_*` ABI

### Lifecycle

- `rr_create` — allocate a CPU vec of `num_envs` and run `my_init` with the kwarg arrays.
- `rr_reset` — reset every env (spawn, odom cache, tyre draw).
- `rr_step` — one control step on every env, OpenMP over the vec.
- `rr_close` — free the vec.
- `rr_set_track` — mmap a different blob into one env; call `rr_reset_env` before the next step.
- `rr_seed` — set one env's RNGs so jitter and tyre draws are reproducible.
- `rr_set_deterministic` — turn sim2real off (call before reset; tyre draw lives in reset).
- `rr_reset_env` — reset one env without latching a terminal (after a track or window change).
- `rr_force_terminal` — latch a collision terminal, log the episode, then reset that env.
- `rr_set_lap_cap` — laps before a complete terminal (floor 1).
- `rr_set_window` — racing-line start index, completion `s_norm` (`-1` = lap-based), and tick cap.

### Buffers

- `rr_observations` — pointer to the flat float obs buffer, `num_envs * rr_obs_size()`.
- `rr_actions` — pointer to the flat float action buffer, two heads per env.
- `rr_rewards` — pointer to the per-env reward buffer.
- `rr_terminals` — pointer to the per-env terminal flag buffer.
- `rr_obs_size` — observation length (layout in `sf/observation.py`).
- `rr_num_atns` — action-head count (steer, speed).

### Per-env state, scans, obstacles

- `rr_current_lap` — live lap index, valid between steps.
- `rr_last_lap_time` — live last flying-lap time in seconds.
- `rr_done_laps` — latched lap index of the episode that just ended.
- `rr_done_lap_time` — latched flying-lap time of that episode.
- `rr_done_kind` — latched class: not-done, collision, or complete.
- `rr_pending_control_period_ms` — ticks `rr_step` will consume for the next action.
- `rr_pending_compute_latency_ms` — ticks the previous command is held at the start of that step.
- `rr_last_compute_latency_ms` — ticks actually held on the step that just ran.
- `rr_n_beams` — native LiDAR beam count for that env.
- `rr_vehicle_state` — true plant state: x, y, yaw, delta, v, r, beta, accel, delta_dot, Fy_f, Fy_r.
- `rr_measured_state` — deploy odom snapshot: x, y, yaw, delta, v, r (no beta; under sim2real x and y each get N(0, `odom_noise_pos_m`) and v gets N(0, `odom_noise_vel_mps`), SI kwargs from `sf/noise.py`, independent of the track).
- `rr_odom_age` — age in seconds of that snapshot at the current scan (scan stamp minus pose stamp on the car).
- `rr_tyre_params` — this episode's mu, S, alpha_pk (drawn at reset when sim2real is on).
- `rr_collision_probe_pose` — signed-EDT centre, min corner, `check_done`, and four corner samples at a pose.
- `rr_raw_scan` — full-resolution raytrace in metres, no sensor noise; beams that hit nothing within `lidar_max_range_m` are `inf`, as the sensor reports them.
- `rr_measured_scan` — the same scan with lidar bias and per-beam noise on the returns, in metres; no-return beams stay `inf`.
- `rr_set_obstacles` — install track-frame rectangles `[s, n, v_s, half_length, half_width, …]` after reset. Rebuilds that env's policy observation at the current pose (same raytrace + construct path as reset, including sim2real obs noise) so the first action sees the boxes.
- `rr_get_obstacles` — read live world pose `[cx, cy, yaw, half_length, half_width, vx, vy, s]`.

### Render overlays

No-ops when `rr_render_available()` is 0.

- `rr_render` — draw one env in a raylib window.
- `rr_render_available` — 1 if this `.so` is linked against raylib.
- `rr_set_det_boxes` — detector boxes `[x, y, yaw, length, width]`; `n=0` clears.
- `rr_set_safety` — chip: 0 clear, 1 filter, 2 unsafe (worse latches, quieter needs a hold).
- `rr_set_trace` — enable or clear the pose trace.
- `rr_push_trace` — append one traced pose and safety mode.
- `rr_set_filter_path` — predicted horizon as packed `[x, y]` metres; `n=0` clears.

Two environment variables save frames from `rr_render`: with `ROBORACER_FRAME_DIR` set (to any
non-empty value), every fifth rendered frame is written as `frame_NNNN.png` in the current working
directory; otherwise `ROBORACER_SCREENSHOT=<path>` saves the first rendered frame to `<path>`.

## Step and timing

The plant integrates at 1 ms (`DT_SIM`). One `rr_step` runs `control_period` of
those ticks. Control period and compute latency come from `vehicle/params.yaml`
(`control_period_s`, `compute_latency_s`), converted to integer millisecond
ticks by the Python side. The command computed from the scan is published after
`compute_latency` ticks of holding the previous command (under `sim2real` the
hold is drawn per decision from N(`compute_latency`, `compute_latency_sd`),
clipped to the period, when `compute_latency_sd_s` is non-zero); it then sits in the
steer and speed delay buffers sized from `steer_transport_s` and
`speed_transport_s` in the same yaml. LiDAR is scan-triggered with the control
period. Odometry is a separate snapshot stream on the `odom_period` kwarg (1 ms
ticks, not a car parameter). When sim2real is on, the odom phase at reset and
the gap between snapshots are jittered; when it is off they are nominal.

## Observation

The observation vector the policy sees is defined by `sf/observation.py`
(odometry, the downsampled LiDAR table, previous action). `rr_obs_size` and the
C obs buffer follow that module. Native scans stay at the beam count in
`vehicle/params.yaml`; only the policy observation is subsampled.

## Obstacles

Obstacles are analytic rectangles in the track frame, not burned into the EDT.
Each has station `s`, lateral offset `n` (left positive), rate `v_s` along the
raceline (zero if parked), and half-length / half-width. After every plant tick
the world pose is rebuilt from the raceline and cached for LiDAR, collision, and
render. `rr_set_obstacles` writes the layout into one env after reset and
rebuilds that env's policy observation at the current pose, so `venv.obs`
includes the boxes before the next step. The env keeps the layout across
in-step auto-resets. A small fixed cap limits how many are live.

## Blob format

`tools/compile_track.py <map>` reads `sf/tracks/<map>/raw/<map>.yaml` (names the
`.pgm`), the occupancy image, and `sf/tracks/<map>/raceline.csv` (header
`x,y,s,s_norm,track_yaw_rad,kappa,w_l,w_r,n_x,n_y`). It writes
`sf/tracks/<map>/<map>.blob`. Occupancy is trinary; unknown cells are walls. The
image is flipped once so the EDT is world-y-up. The field is signed
(`edt(free) - edt(occupied)`), in metres, float32.

Little-endian, 64-byte header, then packed arrays. Magic `RRBC`, version 1.
Header fields at these byte offsets:

| offset | type    | field         |
|--------|---------|---------------|
| 0      | uint32  | magic `RRBC`  |
| 4      | uint32  | version (1)   |
| 8      | int32   | W (cells)     |
| 12     | int32   | H (cells)     |
| 16     | float32 | resolution m/cell |
| 20     | float32 | origin_x      |
| 24     | float32 | origin_y      |
| 28     | int32   | n_rline       |
| 32     | int32   | edt_offset (64) |
| 36     | int32   | rline_offset  |
| 40     | pad     | zeros to 64   |

Then `float32[H * W]` EDT, row-major, world-y-up, signed metres, then
`float32[n_rline * 10]` raceline. The C reader trusts `edt_offset` and
`rline_offset`; it does not infer them from a struct size. World to grid:

```
gx = (wx - origin_x) / resolution
gy = (wy - origin_y) / resolution
```
