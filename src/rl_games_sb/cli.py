"""Command-line interface for the `rlgames-sb` command.

Contents:
  - cmd_*()         : one function per subcommand (version, list, inspect,
                      init, train, tune, delete, load, sim, render)
  - _build_parser() : the argparse parser wiring commands to their options
  - main()          : entry point

Argument parsing and output formatting only; the underlying behaviour lives
in rl_games_sb.registry, rl_games_sb.envs, rl_games_sb.experiments and
Stable-Baselines3.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version

import numpy as np
import yaml
from stable_baselines3.common.evaluation import evaluate_policy

from rl_games_sb import config, envs, experiments, registry
from rl_games_sb.registry import ALGO_CHOICES

ENV_ID = envs.DEFAULT_ENV_ID
VERSION = version("rl_games_sb")


def _fmt(obs: np.ndarray) -> str:
    """A state vector or window in full; an image observation by its shape only."""
    if obs.ndim < 3:
        return np.array2string(obs, precision=3)
    return f"<image {obs.shape} {obs.dtype}>"


# ── commands ─────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    env_id = args.env
    env = envs.make(env_id)

    print(f"Environment: {env_id}\n")
    print(f"Observation space : {env.observation_space}")
    print(f"  shape           : {env.observation_space.shape}")
    print(f"\nAction space      : {env.action_space}")
    if hasattr(env.action_space, "n"):
        print(f"  n actions       : {env.action_space.n}")
    print(f"Max episode steps : {env.spec.max_episode_steps if env.spec else 'N/A'}")

    n = args.steps
    print(f"\n-- Sample transitions ({n} steps, random policy) --\n")
    obs, _ = env.reset()
    print(f"  Initial state: {_fmt(obs)}\n")

    for step in range(1, n + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"  step {step:>3} | action={action} | reward={reward:+.3f} | done={done}")
        print(f"           state -> {_fmt(obs)}")
        if done:
            print("           [episode ended, resetting]")
            obs, _ = env.reset()
            print(f"           state -> {_fmt(obs)}")
        print()

    env.close()


def _hyperparams(args: argparse.Namespace) -> dict:
    """--device as a constructor kwarg, when given."""
    return {"device": args.device} if args.device else {}


def _load(args: argparse.Namespace):
    """Load the saved agent on --device (default: auto) and print the device."""
    model = registry.load(args.agent, args.env, device=args.device or "auto")
    print(f"Device: {registry.describe_device(model.device)}")
    return model


def cmd_init(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)

    if path.exists():
        print(f"Save already exists at {path}. Run 'rlgames-sb delete {args.agent}' first.")
        return

    model = registry.create(args.agent, args.env, **_hyperparams(args))
    model.save(path)
    print(f"Initialized {args.agent} agent at {path}.")
    print(f"Device: {registry.describe_device(model.device)}")


def cmd_train(args: argparse.Namespace) -> None:
    if args.config:
        cfg = config.load(args.config)
        if args.timesteps:
            cfg.timesteps = args.timesteps
        cfg.hyperparams.update(_hyperparams(args))
    elif args.agent:
        cfg = config.Config(
            env=args.env,
            algo=args.agent,
            timesteps=args.timesteps or 100_000,
            experiment=args.env,
            eval_episodes=0,
            hyperparams=_hyperparams(args),
        )
    else:
        raise SystemExit("train: give an agent (e.g. 'train dqn') or --config FILE")

    path = experiments.train(cfg)
    print(f"Training complete. Saved to {path}")


def cmd_tune(args: argparse.Namespace) -> None:
    cfg = config.load(args.config)
    cfg.hyperparams.update(_hyperparams(args))
    study = experiments.tune(cfg, n_trials=args.trials, timesteps=args.timesteps)

    print(f"\nBest mean reward: {study.best_value:.2f} (trial {study.best_trial.number})")
    print("Best hyperparameters, to paste under 'hyperparams:' in the config:\n")
    print(yaml.safe_dump(study.best_params, sort_keys=False))


def cmd_delete(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if path.exists():
        path.unlink()
        print(f"Deleted {path}")
    else:
        print(f"No save found at {path}")


def cmd_load(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if not path.exists():
        print(f"No save found at {path}")
        return

    model = registry.load(args.agent, args.env, device=args.device or "auto")
    params = sum(p.numel() for p in model.policy.parameters())
    print(
        f"{type(model).__name__} agent for {args.env}\n"
        f"  Timesteps trained : {model.num_timesteps:,}\n"
        f"  Policy            : {type(model.policy).__name__}\n"
        f"  Network params    : {params:,}\n"
        f"  LR / Gamma        : {model.learning_rate} / {model.gamma}\n"
        f"  Device            : {registry.describe_device(model.device)}"
    )

    if args.eval:
        print("\nEvaluating (10 episodes) ...")
        mean, std = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"  Mean reward: {mean:.2f} +/- {std:.2f}")


def cmd_sim(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if not path.exists():
        print(f"No save found at {path}")
        return

    model = _load(args)
    env = envs.make(args.env)

    all_rewards: list[float] = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"== Episode {ep}/{args.episodes} ==\n")
        print(f"  initial state: {_fmt(obs)}\n")

        limit = args.steps  # None means show all

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if limit is None or step <= limit:
                print(
                    f"  step {step:>4} | action={action!s:>4} | "
                    f"reward={reward:+8.3f} | total={total_reward:+9.2f}"
                )
                if args.verbose:
                    print(f"           state -> {_fmt(obs)}")

        if limit is not None and step > limit:
            print(f"  ... ({step - limit} more steps) ...")

        outcome = "TRUNCATED (time limit)" if truncated else "TERMINATED"

        print(f"\n  Result: {outcome} | Steps: {step} | Total reward: {total_reward:+.2f}")
        if "total_profit" in info:  # trading envs: profit factor, 1.0 = break even
            print(f"  Total profit: {info['total_profit']:.4f}")
        print()
        all_rewards.append(total_reward)

    env.close()

    if len(all_rewards) > 1:
        print(
            f"Summary over {len(all_rewards)} episodes: "
            f"mean={np.mean(all_rewards):+.2f} +/- {np.std(all_rewards):.2f}"
        )


def cmd_render(args: argparse.Namespace) -> None:
    path = registry.save_path(args.agent, args.env)
    if not path.exists():
        print(f"No save found at {path}")
        return

    model = _load(args)
    env = envs.make(args.env, render_mode="human")

    rewards, _ = evaluate_policy(
        model, env, n_eval_episodes=args.episodes, return_episode_rewards=True
    )
    for ep, total_reward in enumerate(rewards, 1):
        print(f"Episode {ep}/{args.episodes} | Reward: {total_reward:.2f}")

    env.close()


def cmd_version(_args: argparse.Namespace) -> None:
    print(f"rl_games_sb {VERSION}")


def cmd_list(args: argparse.Namespace) -> None:
    print(f"Available agents for {args.env}:\n")
    for algo in ALGO_CHOICES:
        path = registry.save_path(algo, args.env)
        status = "saved" if path.exists() else "no save"
        print(f"  {algo:<14} [{status}]  {path}")


# ── argument parser ──────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames-sb",
        description="Train and evaluate Stable-Baselines3 agents on Gymnasium environments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--env",
            type=str,
            default=ENV_ID,
            help=f"Gymnasium env ID, e.g. ALE/Pong-v5 (default: {ENV_ID})",
        )

    def add_device_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--device",
            choices=("auto", "cpu", "cuda"),
            default=None,
            help="Torch device (default: the config's, else auto = GPU if available)",
        )

    # version
    p = sub.add_parser("version", help="Show the package version")
    p.set_defaults(func=cmd_version)

    # list
    p = sub.add_parser("list", help="List available agents and their save status")
    add_env_arg(p)
    p.set_defaults(func=cmd_list)

    # inspect
    p = sub.add_parser(
        "inspect",
        help="Inspect an environment: show state/action spaces and sample transitions",
    )
    add_env_arg(p)
    p.add_argument("--steps", type=int, default=5, help="Random steps to sample (default: 5)")
    p.set_defaults(func=cmd_inspect)

    # init
    p = sub.add_parser("init", help="Initialize a new (untrained) agent and save it")
    p.add_argument("agent", choices=ALGO_CHOICES)
    add_env_arg(p)
    add_device_arg(p)
    p.set_defaults(func=cmd_init)

    # train
    p = sub.add_parser(
        "train", help="Train an agent and save the result (tracked in MLflow)"
    )
    p.add_argument("agent", nargs="?", choices=ALGO_CHOICES, help="Omit when using --config")
    p.add_argument(
        "--config", type=str, help="Experiment YAML, e.g. configs/breakout_dqn.yaml"
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps (default: the config's, or 100k)",
    )
    add_env_arg(p)
    add_device_arg(p)
    p.set_defaults(func=cmd_train)

    # tune
    p = sub.add_parser(
        "tune", help="Search hyperparameters with Optuna (tracked in MLflow)"
    )
    p.add_argument("--config", type=str, required=True, help="Experiment YAML with a tune: section")
    p.add_argument("--trials", type=int, default=None, help="Number of trials (default: the config's)")
    p.add_argument(
        "--timesteps", type=int, default=None, help="Timesteps per trial (default: the config's)"
    )
    add_device_arg(p)
    p.set_defaults(func=cmd_tune)

    # delete
    p = sub.add_parser("delete", help="Delete a saved agent")
    p.add_argument("agent", choices=ALGO_CHOICES)
    add_env_arg(p)
    p.set_defaults(func=cmd_delete)

    # load
    p = sub.add_parser("load", help="Load a saved agent and display info")
    p.add_argument("agent", choices=ALGO_CHOICES)
    p.add_argument("--eval", action="store_true", help="Run a quick 10-episode evaluation")
    add_env_arg(p)
    add_device_arg(p)
    p.set_defaults(func=cmd_load)

    # sim
    p = sub.add_parser("sim", help="Simulate episodes with a trained agent (text output)")
    p.add_argument("agent", choices=ALGO_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to simulate (default: 1)")
    p.add_argument("--steps", type=int, default=None, help="Limit output to the first N steps per episode (default: show all)")
    p.add_argument("--verbose", action="store_true", help="Print every step with full state vectors")
    add_env_arg(p)
    add_device_arg(p)
    p.set_defaults(func=cmd_sim)

    # render
    p = sub.add_parser("render", help="Render episodes using a saved agent (graphical window)")
    p.add_argument("agent", choices=ALGO_CHOICES)
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to render (default: 1)")
    add_env_arg(p)
    add_device_arg(p)
    p.set_defaults(func=cmd_render)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
