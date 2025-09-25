from types import SimpleNamespace
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from training.utils import build_env, to_scalar_action
from environment import load_grid_config
from expanded_environment import reate_env_from_config as create_expanded_env


def evaluate_model(model, cfg: dict) -> dict:
    """Runs evaluation episodes and computes summary metrics."""

    # Build a single evaluation environment consistent with training wrappers
    episodes = int(cfg.get("eval_episodes", 100))
    env = build_env(SimpleNamespace(**cfg), vec=False)
    # Configure reward regime on the base (unwrapped) environment
    use_combined = cfg.get("reward_mode") == "combined"
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "set_use_combined_rewards"):
        base_env.set_use_combined_rewards(use_combined)
    else:
        setattr(base_env, "use_combined_rewards", use_combined)

    # Per-episode rollouts and metric accumulators
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    try:
        for episode in range(episodes):
            # Use a deterministic seed stream for reproducibility across episodes
            base_seed = cfg.get("eval_seed") or (cfg["seed"] + 999)
            obs, _ = env.reset(seed=int(base_seed) + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            last_info = {}

            while not (terminated or truncated):
                # Greedy (deterministic) action for evaluation
                action, _ = model.predict(
                    obs, deterministic=bool(cfg.get("deterministic_eval", True))
                )
                action = to_scalar_action(action)
                # Step the environment and track reward/length and last info
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if info:
                    last_info = info

            # Append episode metrics
            returns.append(total_reward)
            lengths.append(steps)
            successes.append(last_info.get("reason") == "goal")
    finally:
        # Always close env resources
        env.close()

    # Aggregate summary statistics across episodes
    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    return {
        "episodes": float(len(returns)),
        "average_return": avg_return,
        "average_length": avg_length,
        "success_rate": success_rate,
    }


def online_eval_and_log(model, cfg: dict, trained_ts: int, iter_idx: int) -> float:
    """Runs a lightweight evaluation during training and returns the success rate."""

    # Create a shallow copy config with a small number of episodes
    episodes = max(int(cfg.get("online_eval_episodes", 10)), 1)
    eval_cfg = dict(cfg)
    eval_cfg["eval_episodes"] = episodes
    # Reuse the full evaluator for consistency
    metrics = evaluate_model(model, eval_cfg)

    # Print a compact metric line for logs
    print(
        "[eval] ts={ts} iter={it} episodes={eps:.0f} avg_return={ret:.2f} "
        "avg_length={length:.2f} success_rate={succ:.2%}".format(
            ts=int(trained_ts),
            it=int(iter_idx),
            eps=metrics["episodes"],
            ret=metrics["average_return"],
            length=metrics["average_length"],
            succ=metrics["success_rate"],
        )
    )

    return float(metrics["success_rate"])


def _build_single_env_expanded_eval(config, env_config) -> gym.Env:
    """Creates a single ExpandedGridEnv instance for evaluation.

    Args:
        config: Object exposing use_flatten_wrapper, max_episode_steps, seed or eval_seed.
        env_config: Validated GridConfig used to instantiate the expanded environment.

    Returns:
        gym.Env: Wrapped ExpandedGridEnv instance ready for evaluation.
    """

    # Mirror training-time wrappers
    flatten = bool(getattr(config, "use_flatten_wrapper", False))
    max_steps = getattr(config, "max_episode_steps", None)
    eval_seed = getattr(config, "eval_seed", None)
    base_seed = int(getattr(config, "seed", 0))
    seed = int(eval_seed) if eval_seed is not None else base_seed + 999

    # Instantiate expanded env and apply wrappers + seeding
    env = create_expanded_env(env_config)
    if max_steps is not None:
        env = TimeLimit(env, max_episode_steps=int(max_steps))
    if flatten:
        env = gym.wrappers.FlattenObservation(env)
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    return env


def evaluate_model_expanded(model, cfg: dict) -> dict:
    """Runs evaluation episodes on ExpandedGridEnv and reports summary metrics.

    Args:
        model: Policy with predict that returns an action for a given observation.
        cfg (dict): Keys include eval_episodes, eval_seed, seed, deterministic_eval,
            env_config_path, use_flatten_wrapper, max_episode_steps.

    Returns:
        dict: episodes, average_return, average_length, success_rate,
        average_bonuses_visited across episodes.
    """

    # Build a single expanded evaluation environment
    episodes = int(cfg.get("eval_episodes", 100))
    grid_cfg = load_grid_config(cfg["env_config_path"])
    eval_ns = SimpleNamespace(**cfg)
    env = _build_single_env_expanded_eval(eval_ns, grid_cfg)

    # Accumulators for summary metrics
    returns = []
    lengths = []
    successes = []
    visited_counts = []

    try:
        for episode in range(episodes):
            # Deterministic seed stream for reproducibility
            base_seed = cfg.get("eval_seed") or (cfg["seed"] + 999)
            obs, _ = env.reset(seed=int(base_seed) + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            last_info = {}

            while not (terminated or truncated):
                # Greedy prediction and environment step
                action, _ = model.predict(
                    obs, deterministic=bool(cfg.get("deterministic_eval", True))
                )
                action = to_scalar_action(action)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if info:
                    last_info = info

            # Record per-episode results
            returns.append(total_reward)
            lengths.append(steps)
            successes.append(last_info.get("reason") == "goal")
            visited_counts.append(int(last_info.get("visited", 0)))
    finally:
        # Cleanup environment
        env.close()

    # Aggregate summary statistics across episodes
    avg_return = float(np.mean(returns))
    avg_length = float(np.mean(lengths))
    success_rate = float(np.mean(successes))
    avg_visited = float(np.mean(visited_counts))
    return {
        "episodes": float(len(returns)),
        "average_return": avg_return,
        "average_length": avg_length,
        "success_rate": success_rate,
        "average_bonuses_visited": avg_visited,
    }
