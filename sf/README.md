# sf: predictive safety filter

The safety filter, the obstacle perception it relies on, and the track library, in C++17 with Python
bindings (`ppo_sf_roboracer`). The same module runs in simulation (`evaluation/`) and on the vehicle.
Nothing in `sf/` depends on the rest of the repository except `vehicle/params.h`.

## How the filter works

Every control period (67 ms) the policy proposes a steering angle and a speed. The filter simulates
that command for one period followed by a cautious backup policy (brake to a cruise speed or stop,
and hold a lateral line) over a fixed horizon, using a reduced vehicle model. The command is
certified if this rollout keeps clear of the tracked obstacles and the track walls and ends in a
state the backup can hold indefinitely.

```
policy command ─▶ certified? ── yes ─▶ publish unchanged                     Passthrough
                      │
                      no ─▶ QP: closest certified command                    Qp
                            else a point between QP solution and backup      Segment
                            else the backup command                          Backup
                            else the least-violating command                 Uncertified
```

With an obstacle ahead, the filter certifies a pass on one side (left or right) and keeps that
choice until the obstacle is passed, unless the policy steers clearly to the other side.

## Layout

```
include/, src/
  sf.h, sf.cpp                      SafetyFilter facade: certify(), set_published()
  sf_filter.h, sf_filter.cpp        rollouts, constraint evaluation, candidate search
  sf_model.h, sf_model.cpp          reduced vehicle model used for prediction (RK4)
  sf_barrier.h, sf_barrier.cpp      constraint rows: obstacle, wall, terminal speed and lateral stop
  sf_qp.h, sf_qp.cpp                minimum-norm QP for the corrected command
  sf_geometry.h, sf_rect.h/.cpp     body and obstacle geometry
  sf_math.h                         reproducible trigonometry helpers
  sf_tuning.h                       filter parameters (margins, horizon, operating point)
  track.h, track.cpp                track table, Frenet projection, smoothed curvature
  boundary.h, boundary.cpp          wall points
  obstacle_detector.h/.cpp          LiDAR scan to obstacle candidates
  obstacle_tracker.h/.cpp           obstacle boxes over time
  bindings.cpp                      pybind11 module ppo_sf_roboracer
stack.py                            Python construction of track, detector, tracker and filter
observation.py                      policy observation layout and LiDAR beam table
noise.py                            sensor noise levels used in simulation
tracks/<map>/                       track data (see below)
tests/                              C++ (ctest) and Python tests
externals/                          pybind11, nanoflann, CORE-MATH
```

Vehicle parameters live in `../vehicle/params.h`; filter parameters in `include/sf_tuning.h`.

## Python interface

```python
import ppo_sf_roboracer as sf
import stack  # sf/stack.py

track = sf.TrackHandler()
track.load("sf/tracks/lab-20260906/raceline.csv")
walls = sf.BoundaryIndex(stack.wall_points("lab-20260906"))   # boundary.csv samples
detector = sf.ObstacleDetector(sf.DetectorConfig(), track, walls)
tracker = sf.ObstacleTracker()
safety_filter = sf.SafetyFilter(track, 4.0)       # top speed the certificate is built for [m/s]

obstacles = tracker.update(detector.detect(ranges, lidar_pose))
out = safety_filter.certify(state, obstacles, policy_action)
out.command, out.outcome, out.diagnostics.selection
```

- `certify(state, obstacles, action, include_diagnostics=False)` returns the published command, the
  outcome above and diagnostics (constraint values, number of rollouts, the selected manoeuvre; with
  `include_diagnostics=True` also the predicted trajectory).
- `set_published(action)` records the command actually sent, which the next prediction starts from.
- `out.rl_speed_capped` is set when the policy asked for more than the top speed.

`stack.py` builds the whole stack for a track and is the entry point used by `evaluation/`.

## Track data

`tracks/<map>/` holds one track:

| file | content |
|---|---|
| `raw/`, `clean/` | SLAM occupancy map (`.pgm`, `.yaml`); `clean/` is a hand-cleaned copy used when present |
| `centreline.csv`, `boundary.csv` | centreline with widths; wall points |
| `raceline.csv` | racing line with wall clearance at every station |
| `raceline_mintime.csv` | minimum-time speed profile |
| `track.yaml` | lap length, direction, start pose, minimum lap time |
| `zones.json`, `screening.json` | obstacle placement zones for training and evaluation |

`trackgen/` creates these files from a SLAM map; the simulator's compiled track (`<map>.blob`) is
generated at build time.

## Build and test

```bash
./rebuild                     # from the repository root; or:
cmake -S sf -B sf/build -G Ninja && cmake --build sf/build
ctest --test-dir sf/build
uv run --frozen pytest sf/tests
```

The build is reproducible across compilers and C libraries: FMA contraction and the `sin`/`cos`
builtins are disabled and `atan` uses the correctly rounded CORE-MATH implementation, so an
evaluation gives bit-identical results on another machine.
