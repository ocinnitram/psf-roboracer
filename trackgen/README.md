# trackgen — raw SLAM map to sf/tracks/<map>/

```bash
./track <map> --direction cw|ccw [--start s] [--zones N|a:b,c:d] [--vmax MPS] [--no-plot] [--from <stage>]
```

The launcher pulls the `trackgen` extra (CasADi and friends) itself. Input is the SLAM
output at `sf/tracks/<map>/raw/<map>.{pgm,yaml}`, or the hand-cleaned copy in
`clean/` when that directory exists. The first stdout lines are
the effective settings.

Each stage shows a plot and waits for Enter unless `--no-plot`:

| stage      | plot                                                  |
| ---------- | ----------------------------------------------------- |
| centreline | occupancy, centreline, start                          |
| raceline   | minimum-curvature line + wall clearances              |
| mintime    | velocity over station, racing line on the map         |
| zones      | map with start/end dials per zone; Save writes the file |
| screening  | none                                                  |
| compile    | none (`--no-plot` always)                             |

`--from` restarts at that stage and runs through `track.yaml`. `--start s`
rotates the centreline so station `s` becomes the origin. `--direction`
is the direction the car drives; the traced skeleton is reversed when needed.
`--vmax MPS` is the plant speed cap for the map: it bounds the mintime NLP and
is written to `track.yaml` as `max_speed_mps`. Without it the cap stored in
`track.yaml` is used, else the vehicle's `max_speed_mps` from
`vehicle/params.yaml` (nothing is written). A `--vmax` that differs from the
stored cap is rejected when `--from` is after mintime, since it changes
`opt_time`.

## Stages

1. **centreline** — skeletonise the occupancy image, walk it, smooth it, and
   ray-cast the walls. Writes `centreline.csv` and `boundary.csv`.
2. **raceline** — minimum-curvature line from the centreline. The centreline
   is smoothed onto a 0.25 m grid and its widths are ray-cast from the map
   along the normals. The curvature QP is linearised about its reference, so
   it runs as a trust region: each step shifts the reference by at most
   0.15 m, resamples it on the 0.25 m grid, refits the splines and ray-casts
   the widths again, until the largest remaining shift is at most 0.1 m
   (40 steps at most; the last shift is clipped to 0.15 m too). The line is
   resampled at 5 cm and the wall clearance `w_l`/`w_r` is measured at every
   station by walking the normal at 1 cm until an occupied cell and
   subtracting half a cell, so the clearances never sit outside the mapped
   wall. Inner widths are capped at 0.95 of the radius of curvature. Prints
   each step's shift and `Raceline Length`; the plot title shows a
   curvature-limited lap-time estimate (informational). Writes `raceline.csv`.
3. **mintime** — CasADi/IPOPT minimum-time line in Frenet coordinates of
   `raceline.csv`; needs the raceline stage. The reference is smoothed and
   its widths are ray-cast from the map, as in stage 2. The skeleton
   centreline cannot serve: its hairpin radius is below the half-width, so
   its normals cross inside the track. Speed is bounded by the cap
   (`--vmax`). Writes `raceline_mintime.csv` (includes `v_ref`) and returns
   the lap time stored as `opt_time`.
4. **zones** — obstacle zones are chosen by hand: `--zones N` (default 2) opens
   the map with a start/end dial per zone in raceline metres (station ticks are
   drawn on the line), the green band follows the dials, Save writes
   `zones.json` (`valid_zones_s`). Dials start on the longest straights, or on
   the existing file. Headless: `--zones 5:7.5,16:21` writes those ranges;
   `--no-plot` without ranges keeps the existing file.
5. **screening** — copies those zones into the placement file. Writes
   `screening.json`. There is no screening solver; the lab track's
   zones were set by hand. Numeric screens are left unset (`null`); layouts sample
   `s` then `n` uniformly inside the zones.
6. **compile** — `sim/tools/compile_track.py <map> --no-plot` writes
   `<map>.blob`.
7. **track.yaml** — `map`, `opt_time`, `length`, `direction`, `start_pose`,
   and `max_speed_mps` when a cap is given by `--vmax` or already stored.
   Keys the pipeline does not own (e.g. `filter_v_max_mps`) are carried over
   from the existing file.

`opt_time` is the minimum-time lap from the mintime stage (two decimals) in
the direction given by `--direction`, the number pace and the lap reward are
normalised by. Restarting with `--from` after mintime reuses the value already
in `track.yaml`. The curvature-limited estimate in the raceline plot title is
not it and is never stored.
