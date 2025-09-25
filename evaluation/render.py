import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO

# Ensure project imports resolve when running this script from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment import create_env_from_config, load_grid_config
from training.utils import to_scalar_action

MODEL_CONFIG_FILES = {
    "ppo": "ppo_config.yaml",
    "dqn": "dqn_config.yaml",
}
MODEL_CHECKPOINT_PATTERNS = {
    "ppo": "ppo_model_{reward}.zip",
    "dqn": "dqn_model_{reward}.zip",
}
CONFIG_DIR = REPO_ROOT / "training_configurations"
CHECKPOINT_DIR = REPO_ROOT / "saved_models"


def to_int_tuple(point) -> tuple:
    """grid coordinate into plain Python integer components"""
    # Coerce potential numpy scalars into builtin ints for clean printing
    return (int(point[0]), int(point[1]))


def parse_args() -> argparse.Namespace:
    # Minimal CLI: choose algorithm family and reward shaping mode
    parser = argparse.ArgumentParser(description="Render a saved PPO or DQN agent on the base GridEnv")
    parser.add_argument("--model", choices=("ppo", "dqn"), default="ppo", help="Checkpoint family to load")
    parser.add_argument("--reward", choices=("base", "combined"), default="base", help="Reward shaping mode to apply")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    # Read config YAML and validate it
    raw_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if data is None:
        raise ValueError(f"Configuration file {path} is empty")
    return data


def build_environment(env_config_path: Path, use_flatten: bool, max_steps: int, reward_mode: str) -> gym.Env:
    """Construct the base GridEnv with wrappers mirroring training time settings.

    Args:
        env_config_path (Path): Path to the environment configuration YAML.
        use_flatten (bool): Whether to wrap the environment with observation flattening.
        max_steps int: Maximum episode length before truncation.
        reward_mode (str): Reward regime to activate, either base or combined.

    Returns:
        gym.Env: Wrapped environment ready for evaluation.
    """

    # Create the base env and apply training-time wrappers for shape compatibility
    grid_cfg = load_grid_config(env_config_path)
    env = create_env_from_config(grid_cfg)
    if max_steps is not None:
        env = TimeLimit(env, max_episode_steps=int(max_steps))
    if use_flatten:
        env = gym.wrappers.FlattenObservation(env)

    # Sync reward regime with the checkpoint (base or combined)
    use_combined = reward_mode == "combined"
    base_env = env.unwrapped
    if hasattr(base_env, "set_use_combined_rewards"):
        base_env.set_use_combined_rewards(use_combined)
    else:
        setattr(base_env, "use_combined_rewards", use_combined)
    return env


def render_episode(env: gym.Env, model, deterministic: bool) -> tuple:
    """Roll out one deterministic episode while printing step diagnostics and grid snapshots.

    Args:
        env (gym.Env): Environment instance that exposes GridEnv rendering helpers.
        model: Stable-Baselines3 policy used to generate actions.
        deterministic (bool): Flag indicating whether to request deterministic actions.

    Returns:
        tuple: Episode return, number of steps, termination reason, and visited coordinates.
    """

    # Reset and print the initial state before any action
    obs, info = env.reset()
    base_env = env.unwrapped
    agent_pos = to_int_tuple(base_env.agent_position)
    goal_pos = to_int_tuple(base_env.goal_position)
    obstacles = sorted(to_int_tuple(ob) for ob in base_env.obstacle_positions)

    path = [agent_pos]
    print(f"step=0 | agent={agent_pos} goal={goal_pos} obstacles={obstacles}")
    base_env.render(mode="human")

    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    last_info = info or {}

    # Greedy rollout: predict, step, log, and render until episode ends
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = to_scalar_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if info:
            last_info = info

        base_env = env.unwrapped
        agent_pos = to_int_tuple(base_env.agent_position)
        goal_pos = to_int_tuple(base_env.goal_position)
        obstacles = sorted(to_int_tuple(ob) for ob in base_env.obstacle_positions)
        print(f"step={steps} | agent={agent_pos} goal={goal_pos} obstacles={obstacles}")
        base_env.render(mode="human")
        path.append(agent_pos)

    # Summarize episode results for the caller
    episode_return = float(np.float64(total_reward))
    reason = last_info.get("reason", "unknown")
    return episode_return, steps, reason, path


def main() -> None:
    # Parse CLI and load the corresponding algorithm configuration
    args = parse_args()
    config_name = MODEL_CONFIG_FILES[args.model]
    config_path = CONFIG_DIR / config_name
    config = load_yaml_config(config_path)

    # Recover wrappers and evaluation determinism from the config
    env_config_path = REPO_ROOT / config.get("env_config_path", "training_configurations/env_config.yaml")
    use_flatten = bool(config.get("use_flatten_wrapper", False))
    max_steps_raw = config.get("max_episode_steps")
    max_steps = int(max_steps_raw) if max_steps_raw is not None else None
    deterministic_eval = bool(config.get("deterministic_eval", True))

    # Build the base environment with the correct wrappers and reward mode
    env = build_environment(env_config_path, use_flatten, max_steps, args.reward)

    # Resolve checkpoint path and ensure it exists
    checkpoint_pattern = MODEL_CHECKPOINT_PATTERNS[args.model]
    checkpoint_name = checkpoint_pattern.format(reward=args.reward)
    model_path = CHECKPOINT_DIR / checkpoint_name
    if not model_path.exists():
        raise SystemExit(f"Checkpoint not found: {model_path}")

    # Load the model on CPU for portability, then render one episode
    loader = PPO if args.model == "ppo" else DQN
    model = loader.load(str(model_path), device="cpu")

    try:
        episode_return, steps, reason, path = render_episode(env, model, deterministic_eval)
    finally:
        # Always release resources
        env.close()

    print(f"episode_return={episode_return:.4f} steps={steps} reason={reason}")
    print(f"path={path}")


if __name__ == "__main__":
    main()
