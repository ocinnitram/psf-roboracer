# ppo — training

PPO training on the C plant. Actor mean is the deployed command.

```
train.py        ./train [empty|obstacles|path|run] [--seeds 17,76] [--layout-switch episode|complete|never] [--warmstart <run>] [--base <run>] [--set key=value ...] [--wandb] [--tag t]
agent.py        actor/critic, GAE, CAPS, PPO update, checkpoint save/load
config.py       pydantic schema; only keys train.py reads
layouts.py      track-frame rectangles; sample, load, and save evaluation sets
sweep.py        ./sweep <phase>: random search with hyperband, one seed per trial
select.py       ./select <sweep-id>: rank trials by converged score, write next-stage configs; --seeded ranks over seeds
configs/        empty.yaml, obstacles.yaml (from scratch), obstacles-warmstart.yaml (warm-started from an empty-track run). Map numbers come from sf/tracks/<map>/track.yaml
tests/          config, learn, layout, and sweep checks
```

The thesis policies in `outputs/trained_agents/` are retrained from their stored configurations:

- `./train racing --base racing-agent --seeds 17,76,693`: empty track, 5M steps.
- `./train obstacle --base obstacle-agent --warmstart racing-agent --seeds 17,76,693`: two boxes,
  warm-started from the racing policy of the same seed, 5M steps; an environment keeps its layout until
  `num_laps` are completed (`layout_switch: complete`).

The first argument names the new run. The configs in `configs/` are the starting points for new runs:
`./train` (= `./train empty`) trains on the empty track, `./train obstacles` with two boxes from scratch
(10M steps, layouts redrawn every episode), and `./train obstacles-warmstart --warmstart <run>` warm-starts
from an empty-track run. `./eval <run>` evaluates a result.

In-loop evaluation plants at `EVAL_SEED` (1); the training seed only selects weights and the training plant. `_best_eval` and patience use `score(success_rate, pace)` from `config.py`, the same objective wandb/hyperband/`./select` use.

Outputs: `outputs/trained_agents/<run>/config.yaml`, `_actor_seed_<N>` and `_critic_seed_<N>`, each with `_best_eval` and `_ckpt_<S>` variants.
