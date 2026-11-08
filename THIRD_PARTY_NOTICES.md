# Third-party notices

## Included code

| component | location | license |
|---|---|---|
| [pybind11](https://github.com/pybind/pybind11) v3.1.0 | `sf/externals/pybind11` (git submodule) | BSD-3-Clause |
| [nanoflann](https://github.com/jlblancoc/nanoflann) v1.9.0 | `sf/externals/nanoflann` | BSD-2-Clause, see its `LICENSE` |
| [CORE-MATH](https://core-math.gitlabpages.inria.fr/) correctly rounded `atan`, `atanf`, `atan2f`, `asinf`, `tanf` | `sf/externals/core-math`, `sim/externals/core-math` | MIT, in each file's header |

## Adapted code

| component | location | source | license |
|---|---|---|---|
| minimum-curvature raceline, track preparation and spline helpers | `trackgen/stages/raceline.py` | TUM FTM, [trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers), [global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) | LGPL-3.0 |
| minimum-time raceline | `trackgen/stages/mintime.py` | TUM FTM, [global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization) (`opt_mintime_traj`) | LGPL-3.0 |
| vector-environment harness and C binding structure | `sim/ocean/roboracer/` (`vecenv.h`, `shim.c`, `binding.c`) | [PufferLib](https://github.com/PufferAI/PufferLib), Copyright (c) 2022 PufferAI | MIT |

The two `trackgen` files remain under the GNU LGPL v3.0 (`LICENSES/LGPL-3.0.txt`, which
supplements `LICENSES/GPL-3.0.txt`); the rest of this repository is MIT.

## Ported baselines

The evaluation baselines in `evaluation/` re-implement published planners so they can run in this
simulator. Each module's docstring names the source files and the changes made.

| baseline | source | license |
|---|---|---|
| follow-the-gap (`evaluation/reactive.py`) | B. D. Evans, [f1tenth_benchmarks](https://github.com/BDEvan5/f1tenth_benchmarks) | Apache-2.0 |
| disparity extender (`evaluation/reactive.py`) | UNC disparity extender as carried in [Platooning-F1Tenth](https://github.com/pmusau17/Platooning-F1Tenth) | MIT |
| spline planner, state machine, MAP controller (`evaluation/spliner.py`) | ForzaETH, [race_stack](https://github.com/ForzaETH/race_stack) (`spline_planner`, `state_machine`, `controller` packages) | MIT |

The follow-the-gap port is a modified version of the Apache-2.0 source; the license text is at
<https://www.apache.org/licenses/LICENSE-2.0>.
