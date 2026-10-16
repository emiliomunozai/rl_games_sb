# RL Games (Stable-Baselines3)

The twin of [`rl_games`](https://github.com/emiliomunozai/rl_games), with the
learning algorithms taken from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/). Use it as a
reference baseline: if an SB3 agent solves an env and your hand-written one
does not, the bug is in your agent.

## Setup

```bash
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 1. A single experiment: LunarLander with DQN

An experiment is a YAML file under `configs/` that sets the env, the algorithm,
the timesteps and the hyperparameters.

```bash
rlgames-sb inspect --env LunarLander-v3 --steps 1                # spaces and a sample step
rlgames-sb train   --config configs/lunarlander_dqn.yaml          # train, evaluate, save
rlgames-sb load    dqn --env LunarLander-v3 --eval                # score the saved agent
rlgames-sb render  dqn --env LunarLander-v3 --episodes 3          # watch it land
```

Training ends with a 10-episode evaluation (a mean reward of 200+ counts as
solved) and saves the model to `saves/dqn_LunarLander-v3.zip`. The run is logged
to MLflow with its hyperparameters, learning curve, final score and model:

```bash
mlflow ui          # http://127.0.0.1:5000 -> experiment "lunarlander"
```

Training resumes from an existing save, and the model keeps the hyperparameters
it was saved with. To start over, run `rlgames-sb delete dqn --env LunarLander-v3`.

## 2. Hyperparameter tuning

[Optuna](https://optuna.org/) tries hyperparameter combinations and uses the
earlier results to pick the next ones. The search is defined in the `tune:`
section of the config:

```yaml
tune:
  trials: 20
  timesteps: 100_000          # per seed
  seeds: 2                    # training runs per trial, averaged
  evals: 10                   # evaluations during each run
  eval_episodes: 5            # episodes per evaluation
  score_last: 3               # a run's score: mean of its last 3 evaluations
  search_space:
    learning_rate: {type: float, low: 1.0e-4, high: 1.0e-3, log: true}
    batch_size: {type: categorical, choices: [64, 128, 256]}
    target_update_interval: {type: categorical, choices: [250, 1000, 5000]}
    exploration_fraction: {type: float, low: 0.05, high: 0.3}
```

Write floats as `1.0e-4`, not `1e-4`, because YAML reads the latter as a string.

### Step 1: run the search

```bash
rlgames-sb tune --config configs/lunarlander_dqn.yaml
```

Each trial trains a fresh model per seed and prints one line when it ends.
The end of a run looks like this (your numbers will differ):

```
[I ...] Trial 4 finished with value: 243.25 and parameters: {...}. Best is trial 4 ...
[I ...] Trial 6 pruned.
...
Best mean reward: 251.40 (trial 16)
Best hyperparameters:

learning_rate: 0.000301
batch_size: 128
target_update_interval: 250
exploration_fraction: 0.253

Saved to configs/lunarlander_dqn_tuned.yaml. Train with them:
  rlgames-sb train --config configs/lunarlander_dqn_tuned.yaml
```

RL results vary a lot between runs, so a single run per trial lets a lucky seed
win. That is why each trial trains several seeds (the same ones for every trial)
and scores each by its last few evaluations rather than a single final one.
A trial that falls below the median of earlier trials is stopped early
(`pruned`), which saves time for the promising ones.

### Step 2: browse the results in MLflow

From the project folder:

```bash
mlflow ui          # then open http://127.0.0.1:5000
```

1. Open the experiment **lunarlander**. The tuning run is named
   `tune-dqn-20x100k-<date>-<time>`. Expand it to see one nested run per trial
   (`tune-<date>-<time>-trial-07`).
2. Sort the trials by the `eval/mean_reward` column. Pruned trials have no score
   and carry the tag `pruned = True`.
3. Select several trials and click **Compare**. The parallel coordinates plot
   shows which hyperparameter values go with high rewards.
4. Click a trial to see its learning curves, one per seed: `seed0/eval/mean_reward`,
   `seed1/rollout/ep_rew_mean`, ... Seeds that disagree a lot mean the setting is
   unstable.

The tuning run itself holds the best values (`best.*` params), the
`best_mean_reward` metric, and every tuned config it wrote, as an artifact.

### Step 3: train with the best hyperparameters

The best values are written to `configs/lunarlander_dqn_tuned.yaml`, a copy of
the config with them merged in. Tuning overwrites it each time, and every
version is also kept on its MLflow tuning run.

`train` resumes from an existing save, and a resumed model keeps its old
hyperparameters, so delete the save first:

```bash
rlgames-sb delete dqn --env LunarLander-v3
rlgames-sb train  --config configs/lunarlander_dqn_tuned.yaml
rlgames-sb load   dqn --env LunarLander-v3 --eval
rlgames-sb render dqn --env LunarLander-v3 --episodes 3
```

Trials train for 100k steps, but the full run uses the config's `timesteps`
(200k). Short trials favor settings that learn fast, so check the result. In
MLflow, compare the new `train-dqn-200k-...` run with the one from the original
config: select both and click **Compare**. If the tuned one is not clearly
better, the original hyperparameters were already good. To try again from
the tuned values, tune the tuned config:
`rlgames-sb tune --config configs/lunarlander_dqn_tuned.yaml`.

## 3. Other agents and games

Agents: `dqn`, `ppo`, `a2c`. The policy network is picked from the observation:
`MlpPolicy` for state vectors, `CnnPolicy` for images (Atari). Any Gymnasium env
id works, with or without a config (without one, SB3's defaults are used):

```bash
rlgames-sb train ppo --env CartPole-v1 --timesteps 50000
```

| Config | |
|---|---|
| `configs/lunarlander_dqn.yaml` | DQN on LunarLander |
| `configs/lunarlander_ppo.yaml` | PPO on LunarLander |
| `configs/breakout_dqn.yaml` | DQN on Atari Breakout |

**Atari.** `ALE/...` envs get the standard preprocessing (frame skip, 84x84
grayscale, 4-frame stack). They need millions of timesteps, so use a GPU
(`--device cuda`).

**Trading.** `stocks-v0` and `forex-v0` come from
[gym-anytrading](https://github.com/AminHP/gym-anytrading). The agent trains and
is evaluated on the same price history, so treat it as an RL exercise, not a
strategy.

## 4. How it works

```
configs/                # experiment YAMLs: hyperparameters and search spaces
notebooks/              # step-by-step walkthroughs of a training run
src/rl_games_sb/
├── cli.py              # argument parsing and output formatting only
├── registry.py         # which algorithms exist, where they are saved
├── envs.py             # env construction, incl. Atari and trading envs
├── config.py           # loading experiment YAMLs, sampling search spaces
├── experiments.py      # tracked training and Optuna tuning
└── tracking.py         # MLflow logging
```

`registry.py` maps agent names to SB3 classes and each (agent, env) pair to a
save file, e.g. `saves/dqn_LunarLander-v3.zip`. That file is SB3's own format
(weights, optimizer state, hyperparameters, timestep count), which is what lets
`train` resume. Using it as a library:

```python
from rl_games_sb import registry

model = registry.load_or_create("dqn", "LunarLander-v3")
model.learn(total_timesteps=50_000)
model.save(registry.save_path("dqn", "LunarLander-v3"))
```

## Commands

| Command | What it does |
|---|---|
| `version` | Show the package version |
| `list` | List agents and whether each has a save |
| `inspect` | Show an env's spaces and a few sample steps |
| `init` | Create an untrained agent and save it |
| `train` | Train an agent and save it (logged to MLflow) |
| `tune` | Search hyperparameters with Optuna (logged to MLflow) |
| `delete` | Delete a saved agent |
| `load` | Load a saved agent and show its info (`--eval` to score it) |
| `sim` | Run episodes and print them as text |
| `render` | Run episodes in a window |
