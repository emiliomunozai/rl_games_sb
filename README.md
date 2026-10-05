# RL Games (Stable-Baselines3)

The twin of [`rl_games`](https://github.com/emiliomunozai/rl_games): the same
CLI and project layout, with the learning algorithms taken from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) instead of written
by hand. Use it as a reference baseline: if an SB3 agent solves an env and your
hand-written one does not, the bug is in your agent.

## Agents

| Agent | Algorithm |
|---|---|
| `dqn` | Deep Q-Network |
| `ppo` | Proximal Policy Optimization |
| `a2c` | Advantage Actor-Critic |

The policy network is picked from the observation: `MlpPolicy` for state
vectors (LunarLander, CartPole, ...), `CnnPolicy` for images (Atari).

## Setup

```bash
uv sync
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS
```

### GPU

On Linux and Windows, `uv sync` installs the CUDA 13.0 build of PyTorch
(configured under `[tool.uv.sources]` in `pyproject.toml`). It needs an NVIDIA
driver that supports CUDA 13.0 or newer, and falls back to the CPU on machines
without a GPU. Check it with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Every command that builds or loads a model prints its device, e.g.
`Device: cuda (NVIDIA GeForce RTX 5070 Ti)`. By default it is the GPU if there
is one. `--device cpu` or `--device cuda` overrides it, including a config's
`device:`:

```bash
rlgames-sb train --config configs/lunarlander_ppo.yaml --device cuda
rlgames-sb sim   ppo --env LunarLander-v3 --device cpu
```

The GPU pays off for CNN policies (Atari). Small MLP policies usually train
faster on the CPU, so `configs/lunarlander_ppo.yaml` sets `device: cpu`.

## CLI usage

Same commands as `rlgames`, under `rlgames-sb`. The one difference: `train`
takes `--timesteps` instead of `--episodes`, because SB3 trains by timesteps.

```bash
rlgames-sb version
rlgames-sb list
rlgames-sb inspect --env CartPole-v1 --steps 10
rlgames-sb init   ppo
rlgames-sb train  ppo --timesteps 500000
rlgames-sb train  --config configs/lunarlander_ppo.yaml
rlgames-sb tune   --config configs/lunarlander_ppo.yaml --trials 20
rlgames-sb load   ppo --eval
rlgames-sb sim    ppo --episodes 2 --steps 10 --verbose
rlgames-sb render ppo --episodes 3
rlgames-sb delete ppo
```

## Full example: LunarLander from start to finish

PPO solving LunarLander (mean reward 200+) with the bundled config, tracked in
MLflow. The full run takes about 6 minutes on a CPU.

**1. Look at the environment.** 8 numbers in (position, velocity, angle,
angular velocity, leg contacts), 4 actions out (do nothing, left engine,
main engine, right engine):

```bash
rlgames-sb inspect --env LunarLander-v3 --steps 1
```
```
Observation space : Box([...], (8,), float32)
Action space      : Discrete(4)
Max episode steps : 1000
  Initial state: [-0.004  1.399 -0.424 -0.514  0.005  0.096  0.     0.   ]
  step   1 | action=3 | reward=-0.506 | done=False
```

**2. Train with the config.** `configs/lunarlander_ppo.yaml` sets 1M timesteps
and PPO hyperparameters adapted from RL Baselines3 Zoo:

```bash
rlgames-sb train --config configs/lunarlander_ppo.yaml
```

SB3 prints a progress table every rollout. Watch `ep_rew_mean`: it starts
around -200, crosses 0 at about 120k steps, first reaches 200 at about
530k, and stays there from about 900k. Training
ends with a 10-episode evaluation:

```
Eval (10 episodes): 265.92 +/- 21.82
Training complete. Saved to saves\ppo_LunarLander-v3.zip
```

**3. Look at the run in MLflow.**

```bash
mlflow ui          # http://127.0.0.1:5000 -> experiment "lunarlander"
```

The run has the hyperparameters as params, the learning curve as
`rollout/ep_rew_mean`, the losses under `train/`, the final `eval/mean_reward`,
and the config and model as artifacts.

**4. Evaluate and watch the agent.**

```bash
rlgames-sb load   ppo --env LunarLander-v3 --eval
rlgames-sb sim    ppo --env LunarLander-v3 --steps 5
rlgames-sb render ppo --env LunarLander-v3 --episodes 3   # opens a window
```
```
PPO agent for LunarLander-v3
  Timesteps trained : 1,000,448
  Network params    : 9,797
  Mean reward: 249.62 +/- 44.07
...
  Result: TERMINATED | Steps: 337 | Total reward: +243.66
```

**5. Tune it (optional).** 20 trials of 200k timesteps each, about 25 minutes:

```bash
rlgames-sb tune --config configs/lunarlander_ppo.yaml --trials 20
```

In MLflow the tuning run has one nested run per trial; sort them by
`eval/mean_reward` to compare. Paste the printed best values under
`hyperparams:` in the config, then retrain from scratch (a resumed model keeps
its old hyperparameters):

```bash
rlgames-sb delete ppo --env LunarLander-v3
rlgames-sb train  --config configs/lunarlander_ppo.yaml
```

To see what happens inside `train`, step by step, open
`notebooks/lunarlander_ppo.ipynb`.

## Experiment configs

An experiment is a YAML file under `configs/`: the env, the algorithm, the
timesteps and the hyperparameters, which go straight to the SB3 constructor.
The `tune:` section defines a search space for `rlgames-sb tune`.

```yaml
experiment: breakout          # MLflow experiment name
env: ALE/Breakout-v5
algo: dqn
timesteps: 2_000_000
eval_episodes: 10             # greedy evaluation after training
hyperparams:
  learning_rate: 1.0e-4
  buffer_size: 100_000
tune:
  trials: 10
  timesteps: 200_000          # per trial
  hyperparams:                # overrides for the short trial runs
    learning_starts: 10_000
  search_space:
    learning_rate: {type: float, low: 3.0e-5, high: 1.0e-3, log: true}
    batch_size: {type: categorical, choices: [32, 64]}
```

| Config | |
|---|---|
| `configs/lunarlander_ppo.yaml` | PPO on LunarLander. Trains in minutes, a good first experiment |
| `configs/breakout_dqn.yaml` | DQN on Atari Breakout, adapted from RL Baselines3 Zoo |

`--timesteps` on the command line overrides the config's. Without `--config`,
`train` uses SB3's defaults, as before. Write floats as `1.0e-4`, not `1e-4`,
because YAML reads the latter as a string.

Training resumes from an existing save, and a resumed model keeps the
hyperparameters it was saved with. To train with new hyperparameters, run
`rlgames-sb delete` first.

## Experiment tracking (MLflow)

Every `train` and `tune` run is logged to [MLflow](https://mlflow.org/):

- **params**: algorithm, env, timesteps and the model's hyperparameters
- **metrics**: everything SB3 logs during training (`rollout/ep_rew_mean`,
  `train/loss`, ...) plus `eval/mean_reward` after training
- **artifacts**: the config file and the saved model

```bash
mlflow ui          # then open http://127.0.0.1:5000
```

Runs are stored in `./mlflow.db` and `./mlruns/` (both gitignored). Set
`MLFLOW_TRACKING_URI` to use another store, e.g. a shared MLflow server.

## Hyperparameter tuning

`tune` runs an [Optuna](https://optuna.org/) search over the config's
`search_space`. Each trial trains a fresh model for `tune.timesteps` and is
scored by its mean evaluation reward. In MLflow, a tuning run is one parent run
with a nested run per trial, so trials can be compared side by side.

```bash
rlgames-sb tune --config configs/lunarlander_ppo.yaml --trials 20
```

Trial models are not saved. At the end, `tune` prints the best values as YAML.
Paste them under `hyperparams:` and run `train --config` for the full run.
Short trials favor hyperparameters that learn fast, which are not always the
ones that end up best.

## Notebooks

Each notebook walks through one training run: the environment, a single
`(s, a, r, s', done)` transition, the network, one update computed by hand
(loss and gradient step), a short training run, and evaluation against a
random policy. Open them in VS Code or Jupyter with the project's `.venv` as
the kernel.

| Notebook | |
|---|---|
| `notebooks/lunarlander_ppo.ipynb` | PPO from a state vector: actor-critic, GAE advantages, the clipped loss. Reaches or nears the 200 "solved" mark in about 3 minutes on a CPU |
| `notebooks/breakout_dqn.ipynb` | DQN from pixels: frame stacking, replay buffer, TD targets. Shows the mechanics; playing well needs millions of steps |

## Atari

Any `ALE/...` env id works; there is no Atari-specific code beyond `envs.make()`,
which applies the standard preprocessing (4-frame skip, 84x84 grayscale, a
stack of the last 4 frames):

```bash
rlgames-sb inspect --env ALE/Pong-v5
rlgames-sb train   dqn --env ALE/Pong-v5 --timesteps 2000000
rlgames-sb render  dqn --env ALE/Pong-v5

rlgames-sb train   dqn --env ALE/Breakout-v5 --timesteps 2000000
rlgames-sb render  dqn --env ALE/Breakout-v5
```

Games that wait for FIRE to launch the ball, like Breakout, also get
`FireOnNewLife`, which presses FIRE after a reset and after each lost life.
Without it, an agent that never presses FIRE idles until the time limit.

Atari needs millions of timesteps; train on a GPU if you have one. The DQN replay
buffer is capped at 50k transitions for image envs (`IMAGE_KWARGS` in
`registry.py`), about 2.8 GB of RAM. For tuned hyperparameters per game, see
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).

## Trading

`stocks-v0` and `forex-v0` come from
[gym-anytrading](https://github.com/AminHP/gym-anytrading), using its bundled
datasets (GOOGL daily prices, EUR/USD hourly).

| | |
|---|---|
| Observation | the last 30 (stocks) / 24 (forex) steps of `[price, price change]` |
| Actions | `0` = Sell (go short), `1` = Buy (go long) |
| Reward | price change while holding a long position |
| Episode | one pass over the whole dataset |

```bash
rlgames-sb inspect --env stocks-v0
rlgames-sb train   ppo --env stocks-v0 --timesteps 200000
rlgames-sb sim     ppo --env stocks-v0 --steps 20   # also prints total profit
```

`sim` reports `Total profit` as a factor: `1.0` is break-even, `1.1` is +10%.
The agent trains and is evaluated on the same price history, so a good score
says nothing about future prices. Treat it as an RL exercise, not a strategy.

## Project structure

```
configs/                # experiment YAMLs: hyperparameters and search spaces
notebooks/              # walkthroughs: lunarlander_ppo, breakout_dqn
src/rl_games_sb/
├── cli.py              # argument parsing and output formatting only
├── registry.py         # which algorithms exist, where they are saved
├── envs.py             # env construction, incl. Atari preprocessing and trading envs
├── config.py           # loading experiment YAMLs, sampling search spaces
├── experiments.py      # tracked training and Optuna tuning
└── tracking.py         # MLflow: SB3 logger output, recorded hyperparameters
```

Saves are written to `saves/` in the working directory, one file per
(agent, environment) pair, e.g. `saves/dqn_ALE_Pong-v5.zip`. The `.zip` is
SB3's own format: it bundles the network weights, the optimizer state, the
hyperparameters and the timestep count, which is what lets `train` resume
where it left off. The replay buffer is not included.

## Using it as a library

```python
from rl_games_sb import registry

model = registry.load_or_create("ppo", "CartPole-v1")
model.learn(total_timesteps=50_000)
model.save(registry.save_path("ppo", "CartPole-v1"))
```

Or with a config, tracked in MLflow:

```python
from rl_games_sb import config, experiments

experiments.train(config.load("configs/lunarlander_ppo.yaml"))
```
