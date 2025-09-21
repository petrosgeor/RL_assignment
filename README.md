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
