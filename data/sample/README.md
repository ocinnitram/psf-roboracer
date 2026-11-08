# Sample data

A small sample of the thesis results: one real-vehicle run and the matching simulation runs.
Both use the obstacle policy `obstacle-agent` (seed 17, `best_eval`) on the laboratory track
`lab-20260906`, with the same two boxes: **layout D** of the vehicle experiments, which is
layout `k3` of the evaluation set `evaluation/layouts/car-0908`. On this layout the policy alone
crashes, both on the vehicle and in simulation; with the safety filter it completes every lap.

```
sample/
├── vehicle/20260908_132247_layout-D_filter-on/   real vehicle, filter on, 3/3 laps, no contact
│   ├── trajectory.csv                            ~50 Hz pose and commands
│   ├── lap_times.csv
│   ├── run_info.json
│   └── bag/                                      ROS 2 bag of the run (LiDAR, IMU, odometry, commands)
└── sim/
    ├── car-0908_filter-on/                       the four vehicle layouts in simulation, filter on
    │   ├── summary.json, layouts.tsv
    │   └── filter_log_seed17_k{0..3}.csv/.png    one row per control tick
    └── car-0908_filter-off/                      the same layouts, policy alone
        └── summary.json, layouts.tsv
```

## Vehicle run

Recorded on 2026-09-08 on the 1:10 scale RoboRacer vehicle (Jetson Orin NX Super, SICK TiM571
LiDAR, particle-filter localisation), filter on, three timed laps: 9.70 s, 9.82 s and 8.86 s.
The policy-alone run on the same layout crashed in its first lap.

`trajectory.csv` has one row per localisation update (about 50 Hz). The latest drive command and
controller status are repeated on each row, so rows with the same `k` belong to one 15 Hz control tick.

| column | meaning |
|---|---|
| `t` | time since the logger started [s] |
| `lap` | the controller's lap counter; the final value (3) is the braking after the finish |
| `x`, `y`, `yaw` | localised pose in the map frame [m, m, rad] |
| `steer_cmd`, `speed_cmd` | last published drive command [rad, m/s]; `nan` before the first command |
| `outcome` | filter result for the tick: `Passthrough`, `Qp`, `Segment`, `Backup` or `Uncertified` |
| `sel` | the filter's selection label, e.g. `NoObstacleInHorizon` |
| `obstacles` | number of tracked obstacles |
| `rl_delta`, `rl_v` | command proposed by the policy [rad, m/s] |
| `sf_delta`, `sf_v` | command published after the filter [rad, m/s] |
| `lat` | LiDAR-scan-to-command latency [ms] |
| `ms` | controller computation time [ms] |
| `miss` | cumulative deadline misses |
| `k` | control-tick index |

The first rows are the vehicle waiting at the start line, before the first command.
`lap_times.csv` lists the three timed laps in order; its `lap` labels (2, 3, 4) are offset by the
logger and do not index the timed laps.

### ROS 2 bag

`bag/` is the ROS 2 Humble bag recorded on the vehicle during the run (`rosbag2`, SQLite storage,
53 s, 16,744 messages). `metadata.yaml` lists the topics:

| topic | type | messages | content |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | 802 | SICK TiM571 scans, about 15 Hz |
| `/sensors/imu/raw` | `sensor_msgs/Imu` | 2,668 | VESC IMU |
| `/odom` | `nav_msgs/Odometry` | 2,668 | wheel odometry from the VESC |
| `/pf/pose/odom` | `nav_msgs/Odometry` | 3,268 | particle-filter pose in the map frame |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | 5,936 | transforms (`map`, `odom`, `base_link`, `laser`) |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | 701 | published (filtered) steering and speed |
| `/ppo/status` | `std_msgs/String` | 701 | per-tick controller status: policy and filter command, filter outcome |

Replay it with `ros2 bag play bag/` in a ROS 2 Humble environment. The pose is localised in the
SLAM map `sf/tracks/lab-20260906/raw/lab-20260906.{pgm,yaml}`; the track data derived from it is in
the same directory.

## Simulation runs

Generated with this repository:

```bash
./eval obstacle-agent --sf on  --seeds 17 --eval-set car-0908 --log
./eval obstacle-agent --sf off --seeds 17 --eval-set car-0908
```

The runs are deterministic: these commands reproduce the files in `outputs/eval/`, except for the
measured computation times (`ppo_ms`, `filter_ms`, `certify_ms`).

| layout | policy alone | with safety filter |
|---|---|---|
| A (`k0`) | completed | completed |
| B (`k1`) | completed | completed |
| C (`k2`) | completed | completed |
| D (`k3`) | crashed | completed |

`summary.json` holds the evaluation configuration and per-layout results, and `layouts.tsv` one line
per layout. Each `filter_log_seed17_k<K>.csv` has one row per 15 Hz control tick: the plant state
(`x`, `y`, `yaw`, `v`, ...), the policy command (`rl_*`), the filter command (`sf_*`) and the applied
command (`applied_*`), the filter result (`outcome`, `selection`, `failed`), the detected (`raw_*`) and
tracked (`obs_*`) obstacles, the barrier constraint values (`c_*`), timing (`ppo_ms`, `filter_ms`,
`certify_ms`) and true, estimated and steady-state unmeasured states (`true_*`, `est_*`, `ss_*`).
The column list is `evaluation/log.py`; `evaluation/README.md` describes the timing and filter columns.
