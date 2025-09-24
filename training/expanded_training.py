from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Callable, Optional

import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environment import load_grid_config
from expanded_environment import (
    create_env_from_config as create_expanded_env,
)
from training.utils import to_scalar_action
from training.early_stopping import EarlyStopping


def load_config(path: str | Path) -> dict:
    """Loads a YAML configuration file into a dictionary.

    Args:
        path (str | Path): Location of the training configuration YAML file.

    Returns:
        dict: Parsed configuration values.
    """

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Configuration file {path} is empty")
    return raw


def make_expanded_env_factory(
    env_config,
    *,
    flatten: bool,
    max_steps: Optional[int],
    seed: int,
) -> Callable[[], gym.Env]:
    """Constructs a factory that yields ExpandedGridEnv wrapped as requested.

    Args:
        env_config: Validated GridConfig used by create_expanded_env.
        flatten (bool): When true, apply FlattenObservation to the env.
        max_steps (Optional[int]): If provided, wrap with TimeLimit using this cap.
        seed (int): Base seed for environment and action space seeding.

    Returns:
        Callable[[], gym.Env]: A thunk that creates a freshly wrapped env instance.
    """

    def _init() -> gym.Env:
        env = create_expanded_env(env_config)
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=int(max_steps))
        if flatten:
            env = gym.wrappers.FlattenObservation(env)
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
        return env

    return _init


def build_vec_env_expanded(config, env_config) -> VecEnv:
    """Creates a vectorized ExpandedGridEnv for PPO rollouts.

    Args:
        config: Namespace or mapping exposing n_envs, use_flatten_wrapper,
            max_episode_steps, seed.
        env_config: Validated GridConfig for instantiating ExpandedGridEnv.

    Returns:
        VecEnv: DummyVecEnv hosting multiple ExpandedGridEnv instances.
    """

    n_envs = int(getattr(config, "n_envs", 1))
    flatten = bool(getattr(config, "use_flatten_wrapper", False))
    max_steps = getattr(config, "max_episode_steps", None)
    base_seed = int(getattr(config, "seed", 0))

    env_fns = [
        make_expanded_env_factory(
            env_config,
            flatten=flatten,
            max_steps=max_steps,
            seed=base_seed + idx,
        )
        for idx in range(n_envs)
    ]
    return DummyVecEnv(env_fns)


def build_single_env_expanded(config, env_config) -> gym.Env:
    """Creates a single ExpandedGridEnv for evaluation.

    Args:
        config: Namespace or mapping exposing use_flatten_wrapper,
            max_episode_steps, seed or eval_seed.
        env_config: Validated GridConfig for instantiating ExpandedGridEnv.

    Returns:
        gym.Env: Wrapped ExpandedGridEnv instance.
    """

    flatten = bool(getattr(config, "use_flatten_wrapper", False))
    max_steps = getattr(config, "max_episode_steps", None)
    eval_seed = getattr(config, "eval_seed", None)
    base_seed = int(getattr(config, "seed", 0))
    seed = int(eval_seed) if eval_seed is not None else base_seed

    env_fn = make_expanded_env_factory(
        env_config,
        flatten=flatten,
        max_steps=max_steps,
        seed=seed,
    )
    return env_fn()


def build_model_ppo(cfg: dict, env: VecEnv) -> PPO:
    """Instantiates a PPO model using cfg and a vectorized environment.

    Args:
        cfg (dict): Training configuration with PPO hyperparameters.
        env (VecEnv): Vectorized ExpandedGridEnv for rollouts.

    Returns:
        PPO: Constructed PPO learner on CPU.
    """
    device = 'cpu'
    return PPO(
        policy=cfg["policy"],
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg.get("target_kl"),
        seed=cfg["seed"],
        device=device,
        policy_kwargs=cfg.get("policy_kwargs") or None,
        verbose=0,
    )


def evaluate_model_expanded(model, cfg: dict) -> dict:
    """Runs evaluation episodes on ExpandedGridEnv and reports metrics.

    Args:
        model: SB3 PPO or DQN model implementing predict.
        cfg (dict): Configuration with eval_episodes, eval_seed, seed,
            deterministic_eval, env_config_path, use_flatten_wrapper,
            max_episode_steps.

    Returns:
        dict: Summary with episodes, average_return, average_length,
        success_rate, and average_bonuses_visited across episodes.
    """

    episodes = int(cfg.get("eval_episodes", 100))
    grid_cfg = load_grid_config(cfg["env_config_path"])
    eval_ns = SimpleNamespace(**cfg)
    env = build_single_env_expanded(eval_ns, grid_cfg)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    visited_counts: list[int] = []

    try:
        for episode in range(episodes):
            base_seed = cfg.get("eval_seed") or (cfg["seed"] + 999)
            obs, _ = env.reset(seed=int(base_seed) + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            last_info = {}

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs, deterministic=bool(cfg.get("deterministic_eval", True))
                )
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

    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_visited = float(np.mean(visited_counts)) if visited_counts else 0.0
    return {
        "episodes": float(len(returns)),
        "average_return": avg_return,
        "average_length": avg_length,
        "success_rate": success_rate,
        "average_bonuses_visited": avg_visited,
    }


def online_eval_and_log_expanded(model, cfg: dict, trained_ts: int, iter_idx: int) -> float:
    """Runs a lightweight evaluation and prints a concise summary line.

    Args:
        model: SB3 model implementing predict.
        cfg (dict): Configuration with online_eval_episodes and common eval keys.
        trained_ts (int): Cumulative timesteps trained so far.
        iter_idx (int): Training loop iteration index for logging.

    Returns:
        float: Success rate across the sampled episodes.
    """

    episodes = max(int(cfg.get("online_eval_episodes", 10)), 1)
    eval_cfg = dict(cfg)
    eval_cfg["eval_episodes"] = episodes
    metrics = evaluate_model_expanded(model, eval_cfg)
    print(
        "[eval] ts={ts} iter={it} episodes={eps:.0f} avg_return={ret:.2f} "
        "avg_length={length:.2f} success_rate={succ:.2%} avg_visited={vis:.2f}".format(
            ts=int(trained_ts),
            it=int(iter_idx),
            eps=metrics["episodes"],
            ret=metrics["average_return"],
            length=metrics["average_length"],
            succ=metrics["success_rate"],
            vis=metrics["average_bonuses_visited"],
        )
    )
    return float(metrics["success_rate"])


def train_model_ppo(model: PPO, cfg: dict) -> int:
    """Trains PPO in rollout-sized chunks with periodic eval and early stopping.

    Args:
        model (PPO): PPO model to optimise.
        cfg (dict): Configuration with total_timesteps, n_envs, n_steps,
            eval_every_timesteps, and online_eval_episodes.

    Returns:
        int: Number of training iterations completed.
    """

    total_timesteps = int(cfg["total_timesteps"])
    chunk = max(int(cfg["n_envs"]) * int(cfg["n_steps"]), 1)

    trained = 0
    iterations = 0
    eval_every_ts = int(cfg.get("eval_every_timesteps", 0) or 0)
    next_eval_ts = eval_every_ts if eval_every_ts > 0 else None

    stopper = None
    if eval_every_ts > 0:
        stopper = EarlyStopping(
            patience=5,
            min_delta=0.02,
            verbose=True,
            save_best=True,
            best_path=REPO_ROOT / "saved_models/ppo_model_expanded.zip",
        )
    else:
        print("[early-stop] Disabled (eval_every_timesteps=0)")

    while trained < total_timesteps:
        remain = total_timesteps - trained
        this_chunk = min(chunk, remain)
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=(trained == 0))

        trained += this_chunk
        iterations += 1
        print(f"[ppo] iter={iterations} ts={trained}/{total_timesteps}")

        if next_eval_ts is not None and trained >= next_eval_ts:
            success_rate = online_eval_and_log_expanded(
                model, cfg, trained_ts=trained, iter_idx=iterations
            )
            if stopper and stopper(success_rate, model):
                best = stopper.best()
                best_str = f"{best:.2%}" if best is not None else "n/a"
                print(
                    f"[early-stop] Triggered at iter={iterations} ts={trained} (best success={best_str})"
                )
                break
            next_eval_ts += eval_every_ts

    return iterations


def main() -> None:
    """Train PPO on ExpandedGridEnv and print final evaluation metrics.

    Args:
        None

    Returns:
        None
    """

    # Resolve PPO configuration
    cfg_path = REPO_ROOT / "training_configurations" / "pro_config.yaml"

    # Load algorithm config and environment grid config
    cfg = load_config(cfg_path)
    grid_cfg = load_grid_config(cfg["env_config_path"])  # same grid YAML

    # Construct vectorized env and PPO model, then train with early stopping
    train_env = build_vec_env_expanded(SimpleNamespace(**cfg), grid_cfg)
    try:
        model = build_model_ppo(cfg, train_env)
        _ = train_model_ppo(model, cfg)
    finally:
        train_env.close()

    # Final compact evaluation for visibility (best model already saved)
    metrics = evaluate_model_expanded(model, cfg)
    print(
        "Evaluation complete | episodes: {episodes:.0f} | avg return: {ret:.2f} | "
        "avg length: {length:.2f} | success rate: {succ:.2%} | avg visited: {vis:.2f}".format(
            episodes=metrics["episodes"],
            ret=metrics["average_return"],
            length=metrics["average_length"],
            succ=metrics["success_rate"],
            vis=metrics["average_bonuses_visited"],
        )
    )


if __name__ == "__main__":
    main()
