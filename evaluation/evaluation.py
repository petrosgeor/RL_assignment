"""Standalone evaluator for a saved agent (PPO or DQN).

Runs 100 evaluation episodes using the environment described by the
training_configurations YAML files and prints summary metrics.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO, DQN

# Ensure repository root is on sys.path when running this module directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils import to_scalar_action
from environment import load_grid_config, create_env_from_config
from gymnasium.wrappers import TimeLimit
import gymnasium as gym


def load_config(path: str | Path) -> dict:
    """Loads the configuration YAML as a plain dict.

    Args:
        path (str | Path): Path to the pro configuration YAML file.

    Returns:
        dict: Configuration values used to build the evaluation environment.
    """

    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Configuration file {cfg_path} is empty")
    return raw


def evaluate_model(model: PPO, cfg: dict) -> dict:
    """Runs evaluation episodes and computes summary metrics.

    Args:
        model (PPO): Loaded PPO model used for inference.
        cfg (dict): Configuration dict providing env_config_path, seed, and eval settings.

    Returns:
        dict[str, float]: Episodes evaluated, average_return, average_length, success_rate.
    """

    episodes = int(cfg.get("eval_episodes", 100))
    # Build an evaluation environment without any seeding so layouts are random
    env_cfg_path = REPO_ROOT / cfg.get("env_config_path", "training_configurations/env_config.yaml")
    grid_cfg = load_grid_config(env_cfg_path)
    env = create_env_from_config(grid_cfg)
    if cfg.get("max_episode_steps") is not None:
        env = TimeLimit(env, max_episode_steps=int(cfg["max_episode_steps"]))
    if bool(cfg.get("use_flatten_wrapper", False)):
        env = gym.wrappers.FlattenObservation(env)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    try:
        for episode in range(episodes):
            # No seed is provided here to keep evaluation layouts fully random
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            last_info= {}

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=bool(cfg.get("deterministic_eval", True)))
                action = to_scalar_action(action)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if info:
                    last_info = info

            returns.append(total_reward)
            lengths.append(steps)
            successes.append(last_info.get("reason") == "goal")
    finally:
        env.close()

    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0

    return {
        "episodes": float(len(returns)),
        "average_return": avg_return,
        "average_length": avg_length,
        "success_rate": success_rate,
    }


def main() -> None:
    """CLI entrypoint to evaluate the saved PPO model for 100 episodes.

    Args:
        None

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Evaluate a saved RL agent (PPO or DQN)")
    parser.add_argument(
        "--model",
        choices=("pro", "dqn"),
        default="pro",
        help="Which algorithm to evaluate: 'pro' (PPO) or 'dqn' (DQN)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_name = "pro_config.yaml" if args.model == "pro" else "dqn_config.yaml"
    cfg_path = repo_root / "training_configurations" / cfg_name
    cfg = load_config(cfg_path)
    cfg["eval_episodes"] = 100

    model_path = repo_root / cfg["save_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved SB3 checkpoint not found at {model_path}. Run training first."
        )
    loader = PPO if args.model == "pro" else DQN
    # Always load onto CPU to ensure evaluation runs on CPU
    model = loader.load(str(model_path), device="cpu")

    metrics = evaluate_model(model, cfg)
    print(
        "Standalone evaluation complete | episodes: {episodes:.0f} | avg return: {ret:.2f} | "
        "avg length: {length:.2f} | success rate: {success:.2%}".format(
            episodes=metrics["episodes"],
            ret=metrics["average_return"],
            length=metrics["average_length"],
            success=metrics["success_rate"],
        )
    )


if __name__ == "__main__":
    main()
