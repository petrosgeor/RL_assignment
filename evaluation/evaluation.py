import sys
import argparse
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO, DQN

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils import to_scalar_action
from environment import load_grid_config, create_env_from_config
from expanded_environment import create_env_from_config as create_expanded_env
from gymnasium.wrappers import TimeLimit
import gymnasium as gym


def load_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Configuration file {cfg_path} is empty")
    return raw


def evaluate_model_base(model, cfg: dict) -> dict:
    episodes = int(cfg.get("eval_episodes", 100))
    # Build an evaluation environment without any seeding so layouts are random
    env_cfg_path = REPO_ROOT / cfg.get("env_config_path", "training_configurations/env_config.yaml")
    grid_cfg = load_grid_config(env_cfg_path)
    env = create_env_from_config(grid_cfg)
    if cfg.get("max_episode_steps") is not None:
        env = TimeLimit(env, max_episode_steps=int(cfg["max_episode_steps"]))
    if bool(cfg.get("use_flatten_wrapper", False)):
        env = gym.wrappers.FlattenObservation(env)

    use_combined = cfg.get("reward_mode") == "combined"
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "set_use_combined_rewards"):
        base_env.set_use_combined_rewards(use_combined)
    else:
        setattr(base_env, "use_combined_rewards", use_combined)

    returns = []
    lengths = []
    successes = []

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

    avg_return = float(np.mean(returns))
    avg_length = float(np.mean(lengths))
    success_rate = float(np.mean(successes))

    return {
        "episodes": float(len(returns)),
        "average_return": avg_return,
        "average_length": avg_length,
        "success_rate": success_rate,
    }


def evaluate_model_expanded(model, cfg: dict) -> dict:
    episodes = int(cfg.get("eval_episodes", 100))

    
    env_cfg_path = REPO_ROOT / cfg.get("env_config_path", "training_configurations/env_config.yaml")
    grid_cfg = load_grid_config(env_cfg_path)
    env = create_expanded_env(grid_cfg)
    if cfg.get("max_episode_steps") is not None:
        env = TimeLimit(env, max_episode_steps=int(cfg["max_episode_steps"]))
    if bool(cfg.get("use_flatten_wrapper", False)):
        env = gym.wrappers.FlattenObservation(env)

    returns = []
    lengths = []
    successes = []
    visited_counts = []

    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            last_info = {}

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
            visited_counts.append(int(last_info.get("visited", 0)))
    finally:
        env.close()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved RL agent (PPO or DQN)")
    parser.add_argument(
        "--model",
        choices=("ppo", "dqn"),
        default="ppo",
        help="Which algorithm to evaluate: 'ppo' or 'dqn'",
    )
    parser.add_argument(
        "--env",
        choices=("base", "expanded"),
        default="base",
        help="Which environment to evaluate on: base GridEnv or expanded",
    )
    parser.add_argument(
        "--reward",
        choices=("base", "combined"),
        default="base",
        help="Reward mode for base GridEnv checkpoints",
    )
    args = parser.parse_args()

    # Expanded environment only supports PPO agents.
    if args.env == "expanded" and args.model != "ppo":
        raise SystemExit("Expanded environment evaluation supports PPO only. Use --model ppo.")

    repo_root = Path(__file__).resolve().parents[1]

    if args.env == "expanded":
        cfg_name = "ppo_config.yaml"  # expanded training uses PPO config
    else:
        cfg_name = "ppo_config.yaml" if args.model == "ppo" else "dqn_config.yaml"

    cfg_path = repo_root / "training_configurations" / cfg_name
    cfg = load_config(cfg_path)
    cfg["eval_episodes"] = 100
    if args.env == "base":
        cfg["reward_mode"] = args.reward

    if args.env == "expanded":
        model_path = repo_root / "saved_models" / "ppo_model_expanded.zip"
        loader = PPO
    else:
        if args.model == "ppo":
            model_path = repo_root / "saved_models" / f"ppo_model_{args.reward}.zip"
            loader = PPO
        else:
            model_path = repo_root / "saved_models" / f"dqn_model_{args.reward}.zip"
            loader = DQN

    if not model_path.exists():
        raise FileNotFoundError(f"Saved SB3 checkpoint not found at {model_path}.")

    # Always load onto CPU to ensure evaluation runs on CPU
    model = loader.load(str(model_path), device="cpu")

    if args.env == "base":
        metrics = evaluate_model_base(model, cfg)
        print(
            "Standalone evaluation complete | episodes: {episodes:.0f} | avg return: {ret:.2f} | "
            "avg length: {length:.2f} | success rate: {success:.2%}".format(
                episodes=metrics["episodes"],
                ret=metrics["average_return"],
                length=metrics["average_length"],
                success=metrics["success_rate"],
            )
        )
    else:
        metrics = evaluate_model_expanded(model, cfg)
        print(
            "Standalone evaluation complete | episodes: {episodes:.0f} | avg return: {ret:.2f} | "
            "avg length: {length:.2f} | success rate: {success:.2%} | avg visited: {vis:.2f}".format(
                episodes=metrics["episodes"],
                ret=metrics["average_return"],
                length=metrics["average_length"],
                success=metrics["success_rate"],
                vis=metrics["average_bonuses_visited"],
            )
        )


if __name__ == "__main__":
    main()
