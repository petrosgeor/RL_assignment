# RL Assignment Part 2 Environment

## Configuration File (`training_configurations/env_config.yaml`)
- `grid_rows`: Integer number of rows in the grid (≥ 2).
- `grid_cols`: Integer number of columns in the grid (≥ 2).
- `num_obstacles`: Integer count of obstacles placed per episode (0 ≤ `num_obstacles` ≤ `grid_rows` × `grid_cols` − 2).

### Example
```yaml
grid_rows: 6
grid_cols: 6
num_obstacles: 5
```

## Python API

### Functions

`load_grid_config(path: str | Path = "training_configurations/env_config.yaml") -> GridConfig`
- Loads grid parameters from YAML, validates them, and returns a `GridConfig` instance.
- Example:
  ```python
  from environment import load_grid_config

  config = load_grid_config()
  assert config.grid_rows == 6
  ```

`create_env_from_config(config: GridConfig) -> GridEnv`
- Creates a `GridEnv` instance using the validated configuration values.
- Example:
  ```python
  from environment import create_env_from_config, load_grid_config

  config = load_grid_config()
  env = create_env_from_config(config)
  observation, info = env.reset(seed=42)
  ```

`register_grid_env(env_id: str = "RLAssignment/GridEnv-v0") -> None`
- Registers `GridEnv` with Gymnasium so it can be constructed via `gymnasium.make`.
- Example:
  ```python
  import gymnasium as gym
  from environment import load_grid_config, create_env_from_config, register_grid_env

  register_grid_env()
  env = gym.make("RLAssignment/GridEnv-v0", rows=6, cols=6, num_obstacles=5)
  observation, info = env.reset(seed=0)
  ```

### Classes

`GridConfig`
- **Purpose:** Immutable container for grid dimensions and obstacle count.
- **Attributes:**
  - `grid_rows (int)`: Number of rows in the grid.
  - `grid_cols (int)`: Number of columns in the grid.
  - `num_obstacles (int)`: Number of obstacles to place per episode.
- **Methods:**
  - `validate() -> None`: Ensures the configuration obeys Part 2 constraints.

`GridEnv`
- **Purpose:** Custom Gymnasium environment modeling the randomized grid world.
- **Attributes:**
  - `rows (int)`: Total number of grid rows.
  - `cols (int)`: Total number of grid columns.
  - `num_obstacles (int)`: Number of obstacles enforced after each reset.
  - `action_space (spaces.Discrete)`: Four-directional action encoding (up, down, left, right).
  - `observation_space (spaces.Box)`: `(3, rows, cols)` tensor with agent, goal, and obstacle layers.
  - `agent_position (tuple[int, int])`: Current agent coordinates.
  - `goal_position (tuple[int, int])`: Goal coordinates for the episode.
  - `obstacle_positions (set[tuple[int, int]])`: Set of obstacle coordinates.
- **Methods:**
  - `__init__(rows: int, cols: int, num_obstacles: int) -> None`: Configures grid dimensions and Gym spaces.
  - `reset(*, seed: int | None = None, options: dict[str, Sequence[tuple[int, int]]] | None = None) -> tuple[np.ndarray, dict[str, object]]`: Randomizes episode layout (or applies overrides) and returns the initial observation.
  - `step(action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]`: Applies a discrete action and returns the resulting transition data.
  - `render(mode: str = "human") -> str | None`: Renders the grid for debugging in human or ANSI modes.

### Usage Example
```python
from environment import load_grid_config, create_env_from_config

config = load_grid_config()
env = create_env_from_config(config)
observation, info = env.reset(seed=123)
print("Initial reason:", info["reason"])
terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print("Episode finished due to", info["reason"])
```

## Training (PPO "PRO")

The PPO trainer consumes the environment configuration in
`training_configurations/env_config.yaml` and the training hyperparameters in
`training_configurations/pro_config.yaml`.

Run the trainer (auto-loads the default config and env paths):

```bash
python training/trainer_pro.py
```

You can still target a different configuration with the optional flag:

```bash
python training/trainer_pro.py --config path/to/custom_pro_config.yaml
```

To perform a config-only check, set `dry_run: true` inside your
`pro_config.yaml` before running the script.

Key YAML knobs (excerpt):
- `policy`: `MlpPolicy` (with `use_flatten_wrapper: true`) or `CnnPolicy` (`use_flatten_wrapper: false`).
- `n_envs`, `n_steps`, `batch_size`, `n_epochs`, `learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `vf_coef`, `ent_coef`.
- `eval_every_timesteps`: run periodic online evaluation every N timesteps (0 disables).
- `online_eval_episodes`: number of episodes per periodic evaluation.
- `eval_episodes`: number of episodes for the final post-training evaluation.



## Training (DQN)

A parallel trainer exists for the off-policy DQN baseline. Run it with the
default config:

```bash
python training/trainer_dqn.py
```

The configuration lives in `training_configurations/dqn_config.yaml` and mirrors
PPO's structure (run name, seeds, device, evaluation cadence) while adding DQN
hyperparameters such as `buffer_size`, `train_freq`, `gradient_steps`, and
`target_update_interval`. Periodic and final evaluations use the same metrics
(success rate, average return, average length) for direct comparisons.

## Training Utilities API (training/utils.py)

The helpers below support PPO training and evaluation.

`make_env_factory(env_config: Any, *, flatten: bool, max_steps: Optional[int], seed: int) -> Callable[[], gym.Env]`
- Returns a callable that creates a single GridEnv instance with optional TimeLimit and FlattenObservation wrappers.
- Example:
  ```python
  from types import SimpleNamespace
  from environment import load_grid_config
  from training.utils import make_env_factory

  cfg = load_grid_config("training_configurations/env_config.yaml")
  factory = make_env_factory(cfg, flatten=False, max_steps=None, seed=0)
  env = factory()
  obs, info = env.reset(seed=0)
  ```

`build_training_env(config: Any, env_config: Any) -> VecEnv`
- Builds a DummyVecEnv for PPO rollouts using parameters from config (n_envs, use_flatten_wrapper, max_episode_steps, seed).
- Example:
  ```python
  from types import SimpleNamespace
  from environment import load_grid_config
  from training.utils import build_training_env

  cfg = SimpleNamespace(n_envs=2, use_flatten_wrapper=False, max_episode_steps=None, seed=123)
  env_cfg = load_grid_config()
  vec_env = build_training_env(cfg, env_cfg)
  ```

`build_eval_env(config: Any, env_config: Any) -> gym.Env`
- Creates a single evaluation environment honoring use_flatten_wrapper, max_episode_steps, and eval_seed/seed.
- Example:
  ```python
  from types import SimpleNamespace
  from environment import load_grid_config
  from training.utils import build_eval_env

  cfg = SimpleNamespace(use_flatten_wrapper=False, max_episode_steps=None, seed=0, eval_seed=999)
  env_cfg = load_grid_config()
  eval_env = build_eval_env(cfg, env_cfg)
  ```

`build_env(cfg: Any, *, vec: bool) -> gym.Env | VecEnv`
- Loads cfg.env_config_path and creates either a vectorized training env (vec=True) or a single eval env (vec=False).
- Example:
  ```python
  from types import SimpleNamespace
  from training.utils import build_env

  cfg = SimpleNamespace(
      env_config_path="training_configurations/env_config.yaml",
      n_envs=1,
      use_flatten_wrapper=False,
      max_episode_steps=None,
      seed=0,
  )
  env = build_env(cfg, vec=False)
  ```

`print_summary(config: Any) -> None`
- Prints a concise summary of key PPO settings (run_name, seed, device, total_timesteps, n_envs, policy).
- Example:
  ```python
  from types import SimpleNamespace
  from training.utils import print_summary

  cfg = SimpleNamespace(run_name="demo", seed=0, device="cpu", total_timesteps=1000, n_envs=2, policy="MlpPolicy")
  print_summary(cfg)
  ```

`save_model(model, path: Path) -> Path`
- Saves a Stable-Baselines3 model to path and returns the final resolved path.
- Example:
  ```python
  from pathlib import Path
  from training.utils import save_model

  # model is an SB3 PPO instance
  out = save_model(model, Path("runs/pro_model.zip"))
  ```

`resolve_device(spec: str) -> str`
- Resolves a device string: for auto prefers cuda when available, otherwise cpu.
- Example:
  ```python
  from training.utils import resolve_device
  print(resolve_device("auto"))  # "cuda" if available else "cpu"
  ```

`validate_config(cfg: dict) -> dict`
- Validates required PPO keys, enforces rollout divisibility, checks env_config_path, and fills defaults.
- Example:
  ```python
  from training.utils import validate_config

  cfg = {
      "run_name": "demo",
      "seed": 0,
      "device": "auto",
      "total_timesteps": 1000,
      "save_path": "runs/pro_model.zip",
      "env_config_path": "training_configurations/env_config.yaml",
      "n_envs": 1,
      "use_flatten_wrapper": False,
      "policy": "CnnPolicy",
      "learning_rate": 3e-4,
      "n_steps": 128,
      "batch_size": 64,
      "n_epochs": 4,
      "gamma": 0.99,
      "gae_lambda": 0.95,
      "clip_range": 0.2,
      "ent_coef": 0.0,
      "vf_coef": 0.5,
      "max_grad_norm": 0.5,
  }
  cfg = validate_config(cfg)
  ```

`train_in_chunks(model, total_timesteps: int, *, n_envs: int, n_steps: int) -> int`
- Breaks training into rollout-sized chunks and returns the number of learn iterations performed.
- Example:
  ```python
  from training.utils import train_in_chunks
  # model is an SB3 PPO instance
  iters = train_in_chunks(model, total_timesteps=10_000, n_envs=2, n_steps=128)
  ```

`to_scalar_action(action: Any) -> int`
- Converts SB3 action outputs to a plain Python int (handles numpy scalars and 1-element arrays).
- Example:
  ```python
  import numpy as np
  from training.utils import to_scalar_action
  assert to_scalar_action(np.array(2)) == 2
  ```

## PPO Trainer API (training/trainer_pro.py)

Scriptable helpers and an entrypoint to train and evaluate PPO on GridEnv.

`load_pro_config(path: str | Path) -> dict`
- Loads a YAML PPO config from disk and validates it via validate_config.
- Example:
  ```python
  from training.trainer_pro import load_pro_config
  cfg = load_pro_config("training_configurations/pro_config.yaml")
  ```

`build_model(cfg: dict, env: VecEnv) -> PPO`
- Instantiates a Stable-Baselines3 PPO model using a validated configuration and a vectorized environment.
- Example:
  ```python
  from training.trainer_pro import build_model
  model = build_model(cfg, env)  # env is a VecEnv
  ```

`train_model(model: PPO, cfg: dict) -> int`
- Trains the model in rollout-sized chunks, prints progress, and (optionally) performs periodic evaluations driven by `eval_every_timesteps` and `online_eval_episodes` (printed to stdout).
- Example:
  ```python
  from training.trainer_pro import train_model
  cfg.update({
      "eval_every_timesteps": 4096,
      "online_eval_episodes": 5,
  })
  iterations = train_model(model, cfg)
  ```

`main() -> None`
- CLI entrypoint used by python training/trainer_pro.py to run end-to-end training and evaluation.

## Standalone Evaluation (evaluation/evaluation.py)

`evaluation/evaluation.py`
- Evaluates a saved PPO agent for 100 episodes using the environment from training_configurations.
- Expects an SB3 .zip checkpoint at saved_models/pro_run.zip (falls back to runs/pro_run/models/pro_model.zip) and prints summary metrics.
- Usage:
  ```bash
  conda activate rl_assignment
  python evaluation/evaluation.py
  ```

## Evaluation API (training/training_evaluation.py)

Standalone helpers for evaluation so training and evaluation concerns are separated.

`evaluate_model(model: PPO, cfg: dict) -> dict`
- Runs cfg.eval_episodes on a fresh single env and returns metrics: episodes, average_return, average_length, success_rate.
- Example:
  ```python
  from training.training_evaluation import evaluate_model
  metrics = evaluate_model(model, cfg)
  print(metrics)
  ```

`online_eval_and_log(model: PPO, cfg: dict, trained_ts: int, iter_idx: int) -> None`
- Performs a lightweight evaluation during training using cfg.online_eval_episodes and prints a concise summary line to stdout.
- Example:
  ```python
  from training.training_evaluation import online_eval_and_log
  online_eval_and_log(model, cfg, trained_ts=8192, iter_idx=2)
  ```

## Expanded Training (ExpandedGridEnv)

Train PPO or DQN on the expanded environment with first‑visit bonuses using `training/expanded_training.py`.

Commands:
- PPO: `python training/expanded_training.py --model ppo`
- DQN: `python training/expanded_training.py --model dqn`

The script auto-loads algorithm configs from `training_configurations/pro_config.yaml` or `training_configurations/dqn_config.yaml` and uses the same environment YAML at `training_configurations/env_config.yaml`.

### Expanded Training API (training/expanded_training.py)

`load_config(path: str | Path) -> dict`
- Loads a YAML training configuration into a dictionary.
- Example:
  ```python
  from training.expanded_training import load_config
  cfg = load_config("training_configurations/pro_config.yaml")
  ```

`make_expanded_env_factory(env_config: Any, *, flatten: bool, max_steps: Optional[int], seed: int) -> Callable[[], gym.Env]`
- Returns a thunk that builds an ExpandedGridEnv with optional TimeLimit and FlattenObservation.
- Example:
  ```python
  from types import SimpleNamespace
  from environment import load_grid_config
  from training.expanded_training import make_expanded_env_factory
  env_cfg = load_grid_config()
  factory = make_expanded_env_factory(env_cfg, flatten=True, max_steps=100, seed=0)
  env = factory()
  ```

`build_vec_env_expanded(config: Any, env_config: Any) -> VecEnv`
- Creates a DummyVecEnv of ExpandedGridEnv instances using config.n_envs and wrappers.
- Example:
  ```python
  from types import SimpleNamespace
  from environment import load_grid_config
  from training.expanded_training import build_vec_env_expanded
  cfg = SimpleNamespace(n_envs=2, use_flatten_wrapper=True, max_episode_steps=100, seed=0)
  vec_env = build_vec_env_expanded(cfg, load_grid_config())
  ```

`build_single_env_expanded(config: Any, env_config: Any) -> gym.Env`
- Builds a single ExpandedGridEnv for DQN training or evaluation.
- Example:
  ```python
  from types import SimpleNamespace
  from environment import load_grid_config
  from training.expanded_training import build_single_env_expanded
  cfg = SimpleNamespace(use_flatten_wrapper=True, max_episode_steps=100, seed=0)
  env = build_single_env_expanded(cfg, load_grid_config())
  ```

`build_model_ppo(cfg: dict, env: VecEnv) -> PPO`
- Instantiates PPO on the expanded vectorized environment using keys from pro_config.yaml.
- Example:
  ```python
  from training.expanded_training import build_model_ppo
  model = build_model_ppo(cfg, vec_env)
  ```

`build_model_dqn(cfg: dict, env: gym.Env) -> DQN`
- Instantiates DQN on a single expanded environment using keys from dqn_config.yaml.
- Example:
  ```python
  from training.expanded_training import build_model_dqn
  model = build_model_dqn(cfg, env)
  ```

`evaluate_model_expanded(model, cfg: dict) -> dict`
- Runs cfg.eval_episodes in the expanded env and returns episodes, average_return, average_length, success_rate, and average_bonuses_visited.
- Example:
  ```python
  from training.expanded_training import evaluate_model_expanded
  metrics = evaluate_model_expanded(model, cfg)
  ```

`online_eval_and_log_expanded(model, cfg: dict, trained_ts: int, iter_idx: int) -> float`
- Lightweight periodic evaluation for the expanded env that prints a one-line summary and returns success rate.
- Example:
  ```python
  from training.expanded_training import online_eval_and_log_expanded
  sr = online_eval_and_log_expanded(model, cfg, trained_ts=8192, iter_idx=2)
  ```

`train_model_ppo(model: PPO, cfg: dict) -> int`
- Trains PPO in rollout-sized chunks, optionally evaluating every cfg.eval_every_timesteps.
- Example:
  ```python
  from training.expanded_training import train_model_ppo
  iters = train_model_ppo(model, cfg)
  ```

`train_model_dqn(model: DQN, cfg: dict) -> int`
- Trains DQN in fixed-size chunks, optionally evaluating every cfg.eval_every_timesteps.
- Example:
  ```python
  from training.expanded_training import train_model_dqn
  iters = train_model_dqn(model, cfg)
  ```
