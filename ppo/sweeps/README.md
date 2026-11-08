# Sweeps

Random search on one seed, then the top configs on more seeds, for one training phase
(`empty` or `obstacles`). The search space is `ppo/sweeps/<phase>.yaml`.

## Flow

1. Random search, one seed per trial
   `./sweep empty --agents 4 --trials 50`
2. Rank it and write the next-stage configs
   `./select <sweep-id> --top 5`
   Writes `outputs/sweeps/<sweep-id>.tsv` and `outputs/sweeps/<sweep-id>/<trial>.yaml` for the top 5, and prints one train command per candidate.
3. Run the printed commands, the other two seeds per candidate
   `./train outputs/sweeps/<sweep-id>/<trial>.yaml --seeds 76,693 --wandb --tag seeds-<sweep-id>`
4. Rank the candidates over their seeds
   `./select <sweep-id> --seeded`
   Prints the winning config; its seed runs are in `outputs/trained_agents/<trial>/`.

Obstacles follows the same chain as the warm-start stage: `./sweep obstacles --base obstacles-warmstart --warmstart <empty-winner>` so phase 2 starts from the empty policy (plain `ppo/configs/obstacles.yaml` is the scratch default and is not what this search space was built for). Its search space omits `hidden_size`, `log_std_i`, and the CAPS lambdas: a warmstart pins `hidden_size` to the source run's config and loads its `log_std`, and the CAPS lambdas are held at a fixed fraction (`caps_scale_hold`) rather than tuned. Each trial records its warmstart, and `./select` echoes it into the seed-stage commands: the warm-start source's `_actor_seed_<seed>_best_eval` and `_critic_seed_<seed>_best_eval` must exist for every seed on the machine that runs them (the empty seed stage produces exactly those).

## Score

Every in-loop evaluation logs `score = success_rate + 0.1 * pace`. Completing layouts comes first: a 0.1 gap in success rate beats any pace difference, because a faster crash is still a crash.

A run's score is the **mean of its last K evaluations** (`--last`, default 10 = last 1M steps), not its best evaluation and not its last one: the best is a maximum over noise and the last is one sample of it. A candidate's score is the mean over its seeds; ties go to the worst seed, then to more seeds. Candidates whose score is within 0.005 of the best (a 5 % pace difference, below seed noise) are ordered by mean steer rate, smoothest first: the objective cannot separate them and the car can.

## Commands

```
./sweep <phase> [--base <run>] [--warmstart <run>] [--set k=v ...] [--agents N] [--attach <sweep-id>] [--trials N] [--dry-run]
./select <sweep-id> [--top N] [--last K] [--seeds 76,693] [--base <cfg>] [--set k=v ...] [--seeded]
```

Each agent gets `min(4, (physical_cores - agents) / agents)` torch intra-op threads, printed at launch: the C simulator needs a core per agent and oversubscription costs more throughput than intra-op width buys.

`--base` on `sweep` loads that run's config (or `ppo/configs/<phase>.yaml` when omitted); `--attach` joins an existing wandb sweep instead of creating one; `--dry-run` prints the description and parameter space without contacting wandb; `--agents` processes each pull `--trials` from the shared queue. `select --base` and `--set` repeat what the sweep was launched with, so the rebuilt candidate configs match the trials. Project: `ppo-lab`.
