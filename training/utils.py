import sys
from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Ensure project imports work when running scripts from repo root
    sys.path.insert(0, str(REPO_ROOT))

from environment import create_env_from_config, load_grid_config


def make_env_factory(
    env_config,
    flatten: bool,
    max_steps: int,
    seed: int):
    """Constructs a factory that creates a single wrapped GridEnv instance."""

    def _init() -> gym.Env:
        # Build the base environment from a validated GridConfig
        env = create_env_from_config(env_config)
        # Apply TimeLimit if requested
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_steps)
        # Match training observation shape for MlpPolicy
        if flatten:
            env = gym.wrappers.FlattenObservation(env)
        # Seed environment and action space for reproducibility
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def build_training_env(config, env_config) -> VecEnv:
    """Creates a vectorised environment for PPO rollouts."""

    # Read vectorization and wrapper options from config
    n_envs = int(getattr(config, "n_envs", 1))
    flatten = bool(getattr(config, "use_flatten_wrapper", False))
    max_steps = getattr(config, "max_episode_steps", None)
    base_seed = int(getattr(config, "seed", 0))

    # Create a list of factories with distinct seeds per environment
    env_fns = [
        make_env_factory(
            env_config,
            flatten=flatten,
            max_steps=max_steps,
            seed=base_seed + idx,
        )
        for idx in range(n_envs)
    ]
    # Run environments sequentially in a single process
    return DummyVecEnv(env_fns)


def build_eval_env(config, env_config) -> gym.Env:
    """Creates a single evaluation environment."""

    # Mirror training-time wrappers; prefer a dedicated eval_seed if provided
    flatten = bool(getattr(config, "use_flatten_wrapper", False))
    max_steps = getattr(config, "max_episode_steps", None)
    eval_seed = getattr(config, "eval_seed", None)
    base_seed = int(getattr(config, "seed", 0))
    seed = int(eval_seed) if eval_seed is not None else base_seed + 999

    # Instantiate a single environment using the factory
    env_fn = make_env_factory(
        env_config,
        flatten=flatten,
        max_steps=max_steps,
        seed=seed,
    )
    return env_fn()


def build_env(cfg, vec: bool) -> gym.Env | VecEnv:
    """Builds a training or evaluation environment based on cfg (dict/namespace).

    This helper internally loads the grid config from cfg.env_config_path and
    applies the same wrappers as build_training_env/build_eval_env.
    """

    # Resolve env_config_path relative to repo root and load GridConfig
    env_cfg_path = Path(getattr(cfg, "env_config_path"))
    if not env_cfg_path.is_absolute():
        env_cfg_path = REPO_ROOT / env_cfg_path
    grid_cfg = load_grid_config(env_cfg_path)

    # Dispatch to vectorized or single-environment builder
    if vec:
        return build_training_env(cfg, grid_cfg)
    return build_eval_env(cfg, grid_cfg)







def print_summary(config) -> None:
    """Prints a concise run summary for the provided configuration."""

    # Collect a small set of commonly inspected hyperparameters
    summary = {
        "run_name": getattr(config, "run_name", ""),
        "seed": getattr(config, "seed", ""),
        "device": getattr(config, "device", ""),
        "total_timesteps": getattr(config, "total_timesteps", ""),
        "n_envs": getattr(config, "n_envs", ""),
        "policy": getattr(config, "policy", ""),
    }
    # Pretty-print summary for quick inspection in logs
    print("Training Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")




def save_model(model, path: Path) -> Path:
    """Saves the model to disk and returns the path."""

    # Ensure parent directory exists and persist the SB3 model
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path





def train_in_chunks(model, total_timesteps: int, n_envs: int, n_steps: int) -> int:
    """Trains using rollout-sized chunks and returns the number of iterations."""

    # Use rollout-sized updates when possible (n_envs * n_steps)
    chunk = max(int(n_envs) * int(n_steps), 1)
    trained = 0
    iterations = 0
    while trained < int(total_timesteps):
        # Bound the last chunk to the remaining timesteps
        remain = int(total_timesteps) - trained
        this_chunk = min(chunk, remain)
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=(trained == 0))
        trained += this_chunk
        iterations += 1
    return iterations


def to_scalar_action(action) -> int:
    """Converts SB3 action to a Python int for gym.Env.step.

    Handles numpy arrays of shape () or (1,).
    """
    # SB3 may return numpy scalars/arrays
    if isinstance(action, np.ndarray):
        return int(np.asarray(action).item())
    return int(action)
