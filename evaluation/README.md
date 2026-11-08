# evaluation

One command: `./eval [run (default racing-agent)] [--sf on|shadow|off] [--seeds 17,76] [--checkpoint best_eval|final|<step>] [--eval-set NAME|empty] [--map MAP] [--layout k|all] [--laps N] [--sim2real on|off] [--obstacles detector|truth] [--jobs N] [--log] [--render] [--slowdown X] [--driver policy|ftg|disparity|spliner|spliner-pp] [--speed-scale X]`.

```
cli.py       argument parsing; first stdout line is the effective eval config
policy.py    --sf off (no --render): batched policy evaluation
filter.py    --sf on|shadow, or --sf off --render: tick loop
log.py       filter_log.csv, one row per tick (--log)
reactive.py  --driver ftg|disparity: follow-the-gap (Evans) and the disparity extender (Otterness) on the scan
spliner.py   --driver spliner: ForzaETH's evasion splines, state machine and MAP controller on the tracked obstacles
report.py    summary.json, layouts.tsv, printed table, filter-log plots
common.py    shared paths, plant seed, LayoutResult
layouts/     frozen eval sets, all on map lab-20260906 (`obs-set-dev-500` is the 500-layout set); generate.py --map <map> builds a new one
tools/       map_lut.cpp (MAP steering lookup from the filter model); reactive_crashes.py (crash position and box/wall per crashed layout of a logged run); measure_tracker_noise.py; prediction_error.py (accuracy of the filter's rollout prediction against the plant and the car); observer_error.py (unmeasured-state estimate vs plant, from --log true_/est_/ss_ columns)
tests/
```

`--jobs` fans layouts out over processes in the tick loop. `--render` opens a live window. `--sf shadow` runs the filter and logs it but applies the policy command; after each tick the filter is told that applied command (`set_published`) so the next `u_prev` is what the actuators held.

Layout outcomes: `completed` (plant `COMPLETE`, or every requested flying lap), `crashed` (plant `COLLISION`), `parked` (standstill for 8 s after first moving; tick loop only), `timeout` (plant tick cap, map-scaled: `ppo.rollout.episode_tick_cap`). `--sf off` (batched, no `--render`) uses the same plant kinds without the parked rule.

Outputs: `outputs/eval/<run>/<YYYYmmdd-HHMMSS>_sf-<mode>_<eval_set>[_on-<map>]/` with `summary.json`, `layouts.tsv`, and when `--log` a `filter_log_seed<S>_k<K>.csv` plus `.png`.

`--map <map>` evaluates the run's policy on another map (cross-map; the `_on-<map>` suffix marks it). Track numbers come from that map's `track.yaml` (length for `s_norm`, `opt_time` for the tick cap and pace, start pose) and so does the filter's `filter_v_max_mps` (vehicle cap when unset); the plant speed cap stays the policy's own trained one, because it scales the policy's speed action and speed observation. The eval set defaults to `empty` and must belong to `<map>`. `summary.json` records `eval.map` and `eval.trained_map`.

Per-tick filter columns: `filter_ms` (perception plus certify), `certify_ms` (`SafetyFilter.certify` alone), `n_sims` (rollouts evaluated that period), `failed` (most violated row of the published rollout, `ok` when all hold); empty when the filter does not run.

`--driver ftg|disparity` replaces the policy with a reactive baseline that reads the scan the detector reads (measured under sim2real) and nothing else; its command is logged as `rl_*` and goes through the filter when `--sf on|shadow`. It always uses the tick loop, the run supplies only the map, eval set and plant, and one seed labels the run. `--speed-scale` multiplies the baseline's speed settings (follow-the-gap: Evans' 5/5/3 m/s; disparity extender: the vehicle's top speed). Out dir: `..._sf-<mode>_<driver>-x<scale>_<eval_set>`. Ports and the one lab-track change to follow-the-gap: `reactive.py` docstring.

`--driver spliner` (`spliner-pp`: the same with pure pursuit instead of MAP) is ForzaETH's head-to-head planner instead: it builds the detector and tracker even with `--sf off`, converts the tracked boxes to Frenet coordinates of the raceline, splines around the closest one and tracks the path with ForzaETH's MAP controller, whose steering lookup `data/map_lut.csv` comes from the identified model (`tools/map_lut.cpp`; `spliner.py` docstring: what is ported, what is left out, and the lab-track parameters). `--speed-scale` scales its speed profile (1 = the vehicle's lateral and longitudinal acceleration limits). The state (GB_TRACK, TRAILING, OVERTAKE) is logged as `selection` and a rejected spline's reason as `failed`.
