"""Tracked training and hyperparameter tuning.

Contents:
  - Progress : callback printing one progress line every 5% of training
  - train()  : train (or resume) an agent as one MLflow run
  - tune()   : an Optuna search over a config's search space; one parent MLflow
               run, one nested run per trial

Run names carry a timestamp so repeated runs are told apart in MLflow, e.g.
train-dqn-200k-1016-1342, tune-dqn-20x100k-1016-1342, tune-1016-1342-trial-07.

Both take a config.Config, built from a YAML file or from CLI arguments.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rl_games_sb import config, envs, registry, tracking


class Progress(BaseCallback):
    """Print one line about every 5% of training: steps, mean reward, speed.

    The full SB3 metrics (losses, exploration rate, ...) go to MLflow instead.
    """

    def __init__(self, total: int, every: float = 0.05):
        super().__init__()
        self.total = total
        self.interval = max(1, int(total * every))

    def _on_training_start(self) -> None:
        self.start_step = self.num_timesteps
        self.start_time = time.time()
        self.next_print = self.interval

    def _on_step(self) -> bool:
        done = self.num_timesteps - self.start_step
        if done >= self.next_print or done >= self.total:
            self.next_print += self.interval
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            reward = f"{np.mean(rewards):7.1f}" if rewards else "      -"
            fps = done / max(time.time() - self.start_time, 1e-9)
            print(
                f"  {done:>{len(f'{self.total:,}')},} / {self.total:,} steps "
                f"({done / self.total:4.0%}) | reward {reward} | {fps:,.0f} steps/s"
            )
        return True


def _short(n: int) -> str:
    """200_000 -> "200k", 1_500_000 -> "1.5M"."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    return f"{n / 1_000:g}k" if n >= 1_000 else str(n)


def _stamp() -> str:
    return time.strftime("%m%d-%H%M")


def train(cfg: config.Config) -> Path:
    """Train for cfg.timesteps, save, evaluate, and log it all to MLflow."""
    path = registry.save_path(cfg.algo, cfg.env)
    resumed = path.exists()
    if resumed and cfg.hyperparams.keys() - {"device"}:
        print(
            f"Resuming {path}: it keeps the hyperparameters it was saved with, "
            f"so the config's are ignored. Run 'rlgames-sb delete {cfg.algo} "
            f"--env {cfg.env}' to start fresh."
        )
    model = registry.load_or_create(cfg.algo, cfg.env, **cfg.hyperparams)
    print(f"Device: {registry.describe_device(model.device)}")

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=f"train-{cfg.algo}-{_short(cfg.timesteps)}-{_stamp()}"):
        mlflow.set_tags({"algo": cfg.algo, "env": cfg.env, "resumed": resumed})
        mlflow.log_params({
            "algo": cfg.algo,
            "env": cfg.env,
            "timesteps": cfg.timesteps,
            "start_timesteps": model.num_timesteps,
            **tracking.model_params(model),
        })
        if cfg.path:
            mlflow.log_artifact(str(cfg.path))

        tracking.attach(model, stdout=False)
        print(
            f"Training {cfg.algo.upper()} on {cfg.env} for {cfg.timesteps:,} steps "
            f"(reward = mean over the last 100 episodes)"
        )
        model.learn(
            total_timesteps=cfg.timesteps,
            reset_num_timesteps=False,
            callback=Progress(cfg.timesteps),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        mlflow.log_artifact(str(path))

        if cfg.eval_episodes:
            mean, std = evaluate_policy(
                model, envs.make(cfg.env), n_eval_episodes=cfg.eval_episodes
            )
            mlflow.log_metrics(
                {"eval/mean_reward": mean, "eval/std_reward": std}, step=model.num_timesteps
            )
            print(f"Eval ({cfg.eval_episodes} episodes): {mean:.2f} +/- {std:.2f}")

    return path


class _TrialEval(BaseCallback):
    """Evaluate every `every` steps during a tuning trial.

    Reports the mean of the last `last` evaluations to Optuna (so one lucky
    or unlucky evaluation doesn't decide a trial) and raises TrialPruned when
    the pruner says this trial is worse than earlier ones at the same point.
    """

    def __init__(
        self, trial: optuna.Trial, eval_env, every: int, episodes: int,
        last: int, step_offset: int, prefix: str,
    ):
        super().__init__()
        self.trial, self.eval_env = trial, eval_env
        self.every, self.episodes, self.last = every, episodes, last
        self.step_offset, self.prefix = step_offset, prefix
        self.scores: list[float] = []

    @property
    def score(self) -> float:
        return float(np.mean(self.scores[-self.last:]))

    def _on_step(self) -> bool:
        if self.n_calls % self.every:
            return True
        mean, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.episodes)
        self.scores.append(mean)
        mlflow.log_metric(f"{self.prefix}eval/mean_reward", mean, step=self.num_timesteps)
        self.trial.report(self.score, self.step_offset + len(self.scores))
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def tune(
    cfg: config.Config, n_trials: int | None = None, timesteps: int | None = None
) -> tuple[optuna.Study, Path]:
    """Search cfg.tune.search_space, maximizing mean evaluation reward.

    Each trial trains `tune.seeds` fresh models (seeds 0, 1, ..., the same for
    every trial) for `timesteps` each (default: tune.timesteps). A seed is
    evaluated `tune.evals` times during training, with `tune.eval_episodes`
    episodes each, and scored by the mean of its last `tune.score_last`
    evaluations. The trial's score is the mean over its seeds. Averaging over
    seeds and evaluations keeps one lucky run from winning the search.

    Optuna's MedianPruner stops a trial early when its score is below the
    median of earlier trials at the same point of training.

    Hyperparameters are cfg.hyperparams, overridden by tune.hyperparams
    (settings for short trial runs), overridden by the sampled values. Trial
    models are not saved. The best values are written to a "<config>_tuned.yaml"
    next to the config, ready for `train --config`; its path is returned.
    """
    space = cfg.tune.get("search_space")
    if not space:
        raise ValueError(f"{cfg.path}: no tune.search_space to search")
    n_trials = n_trials or int(cfg.tune.get("trials", 20))
    timesteps = timesteps or int(cfg.tune.get("timesteps", cfg.timesteps))
    seeds = int(cfg.tune.get("seeds", 2))
    n_evals = int(cfg.tune.get("evals", 10))
    eval_episodes = int(cfg.tune.get("eval_episodes", 5))
    score_last = int(cfg.tune.get("score_last", 3))
    every = max(1, timesteps // n_evals)
    base = {**cfg.hyperparams, **(cfg.tune.get("hyperparams") or {})}
    print(f"Device: {registry.describe_device(base.get('device', 'auto'))}")
    print(
        f"Tuning {cfg.algo.upper()} on {cfg.env}: {n_trials} trials x {seeds} seeds "
        f"x {timesteps:,} steps, scored by the mean of the last {score_last} of "
        f"{n_evals} evaluations"
    )

    stamp = _stamp()
    mlflow.set_experiment(cfg.experiment)
    run_name = f"tune-{cfg.algo}-{n_trials}x{_short(timesteps)}-{stamp}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"algo": cfg.algo, "env": cfg.env, "kind": "tune"})
        mlflow.log_params({
            "algo": cfg.algo,
            "env": cfg.env,
            "trials": n_trials,
            "timesteps_per_trial": timesteps,
            "seeds": seeds,
            "evals": n_evals,
            "eval_episodes": eval_episodes,
            "score_last": score_last,
        })
        if cfg.path:
            mlflow.log_artifact(str(cfg.path))

        def objective(trial: optuna.Trial) -> float:
            sampled = config.sample(trial, space)
            name = f"tune-{stamp}-trial-{trial.number:02d}"
            pruned = False
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"algo": cfg.algo, "env": cfg.env, "kind": "trial"})
                mlflow.log_params(sampled)
                scores = []
                try:
                    for seed in range(seeds):
                        model = registry.create(
                            cfg.algo, cfg.env, **{**base, **sampled, "seed": seed, "verbose": 0}
                        )
                        prefix = f"seed{seed}/"
                        tracking.attach(model, stdout=False, prefix=prefix)
                        evaluator = _TrialEval(
                            trial, envs.make(cfg.env), every, eval_episodes,
                            score_last, step_offset=seed * n_evals, prefix=prefix,
                        )
                        model.learn(total_timesteps=timesteps, callback=evaluator)
                        scores.append(evaluator.score)
                        mlflow.log_metric(f"{prefix}score", evaluator.score)
                except optuna.TrialPruned:
                    pruned = True
                mlflow.set_tag("pruned", pruned)
                if not pruned:
                    mlflow.log_metrics({
                        "eval/mean_reward": float(np.mean(scores)),
                        "eval/std_reward": float(np.std(scores)),
                    })
            if pruned:
                raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            # Don't prune before 5 trials have finished, or in the first half
            # of a trial's first seed: early rewards say little.
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=n_evals // 2
            ),
        )
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_mean_reward", study.best_value)
        tuned = config.save_tuned(cfg, study.best_params, study.best_value, run_name)
        mlflow.log_artifact(str(tuned))

    return study, tuned
