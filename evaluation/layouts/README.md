# Evaluation layouts

Frozen obstacle layouts used by training eval and `./eval`. Each set is a
directory of JSON manifests plus a contact sheet PNG (gitignored).

## Manifest

`evaluation/layouts/<set>/k<index>.json`:

```json
{
  "map": "lab-20260906",
  "set": "obs-set-dev-500",
  "index": 0,
  "obstacles": [
    {
      "s": 6.85,
      "n": 0.385,
      "half_length": 0.175,
      "half_width": 0.125,
      "x": 6.927,
      "y": 0.083,
      "yaw": -0.408
    }
  ]
}
```

`s, n, half_length, half_width` go to the simulator. `x, y, yaw` are the
world pose of the same rectangle (track heading at `s`) and are for plots.

## Sets

| set | map | layouts | seed | role |
|---|---|---|---|---|
| `obs-set-dev-500` | `lab-20260906` | 500 | 500 | in-loop eval and `./eval` |
| `obs-set-tune-50` | `lab-20260906` | 50 | 4242 | tuning the spliner baseline (`evaluation/spliner.py`), kept apart from `obs-set-dev-500` |
| `car-0908` | `lab-20260906` | 4 | - | the 2026-09-08 vehicle crate placements, for sim/car comparison |
| `ood-apex` | `lab-20260906` | 2 | - | single crate outside the training zones, from vehicle runs |

`obs-set-dev-500`: two boxes, `min_s_distance` 5.0 m, zones from `sf/tracks/lab-20260906/screening.json`.
To reproduce a set byte-for-byte, move it out of `evaluation/layouts/` first: disjointness
(below) otherwise excludes every one of its own layouts.
`car-0908` and `ood-apex` are measured, not generated: each vehicle run's recorded `/scan` was
replayed through the detector/tracker to get the crate box centres in the map frame, and the
simulator's raceline placement (`place_obstacle_in_world`) was inverted to get `s, n`. Boxes are the nominal 0.35 x 0.25 m crate, long side
along the track unless the fitted box recovered the full crate and lies across it.

| set | k | layout | crates `(s, n)` [m] |
|---|---|---|---|
| `car-0908` | 0 | A | (6.85, -0.12), (20.28, -0.39) |
| `car-0908` | 1 | B | (4.85, -0.23), (15.84, -0.31) |
| `car-0908` | 2 | C | (5.82, -0.01), (18.37, +0.01) |
| `car-0908` | 3 | D | (7.36, +0.13), (15.58, +0.07) |
| `ood-apex` | 0 | - | (12.98, +0.48), crosswise |
| `ood-apex` | 1 | - | (21.72, -0.02) |

The `ood-apex` layouts are single-crate placements outside the training zones, reported separately in
the thesis appendix; on the vehicle, k0 crashed with the filter on. Both stations lie outside the zones
(`zones.json` clipped by `start_exclusion` 5.0 m: 5.0-7.2 and 15.5-21.0 m). k0 is in the hairpin
(curvature -0.62 1/m), 0.48 m left of the raceline; k1 sits on the raceline at the tightest corner
(-0.91 1/m). `car-0908` k1 crate 0 (s 4.85) and k3 crate 0 (s 7.36) also fall just outside the zones.

## Generate

```bash
uv run --frozen python evaluation/layouts/generate.py \
  --set obs-set-dev-500 --layouts 500 --seed 500 --map lab-20260906
uv run --frozen python evaluation/layouts/generate.py \
  --set obs-set-tune-50 --layouts 50 --seed 4242 --map lab-20260906
```

Reads `sf/tracks/<map>/screening.json` and `sf/tracks/<map>/raceline.csv`.
Writes `evaluation/layouts/<set>/k<i>.json` and `<set>.png`. Refuses to
overwrite a set that already has manifests.

Stations and laterals are enumerated on a grid (`s_step_m`, `n_step_m`),
clipped to `valid_zones_s` and `start_exclusion`, then screened on curvature,
width, and width variation when those keys are not null. An empty
`visibility_approach_samples_m` list skips the sight check. `extra_reserve_m`
null accepts every pair that fits and samples stations then `n` uniformly;
a non-null `main_mix` uses the one-sided / two-sided class mix.

## Disjointness

A layout is kept only when its rounded `(s, n)` sequence (three decimals,
obstacles in file order) is not already present in **any** set under
`evaluation/layouts/`. Generate new sets after the ones they must stay
clear of.
