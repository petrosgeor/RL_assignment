"""Grid-based Gymnasium environment for Part 2 of the RL assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

Action = int
Coordinate = Tuple[int, int]

# Mapping from discrete action index to row/column deltas.
ACTION_TO_DELTA: dict[Action, Coordinate] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


@dataclass(frozen=True)
class GridConfig:
    """Immortalizes the grid dimensions and obstacle count for the environment.

    Args:
        grid_rows (int): Number of rows ``M`` in the grid.
        grid_cols (int): Number of columns ``N`` in the grid.
        num_obstacles (int): Number of static obstacles ``K`` to place each episode.

    Attributes:
        grid_rows (int): Stored number of grid rows.
        grid_cols (int): Stored number of grid columns.
        num_obstacles (int): Stored number of obstacles enforced at reset.
    """

    grid_rows: int
    grid_cols: int
    num_obstacles: int

    def validate(self) -> None:
        """Checks that the configuration satisfies the Part 2 feasibility constraints.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: If any dimension is invalid or the obstacle count is infeasible.
        """

        if self.grid_rows < 2 or self.grid_cols < 2:
            raise ValueError("grid_rows and grid_cols must both be >= 2")
        max_obstacles = self.grid_rows * self.grid_cols - 2
        if self.num_obstacles < 0 or self.num_obstacles > max_obstacles:
            msg = (
                "num_obstacles must be between 0 and grid_rows * grid_cols - 2 "
                f"(received {self.num_obstacles}, limit {max_obstacles})"
            )
            raise ValueError(msg)


def load_grid_config(
    path: str | Path = "training_configurations/env_config.yaml",
) -> GridConfig:
    """Loads and validates the grid configuration stored in YAML format.

    Args:
        path (str | Path): Location of the YAML file describing the grid. By
            default, this reads from ``training_configurations/env_config.yaml``.
            For backward compatibility, if the file is not found at the given
            path and it equals the default, the loader will fall back to
            ``config.yaml`` in the project root.

    Returns:
        GridConfig: Validated configuration ready for environment construction.
    """

    cfg_path = Path(path)

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = GridConfig(
        grid_rows=int(raw["grid_rows"]),
        grid_cols=int(raw["grid_cols"]),
        num_obstacles=int(raw["num_obstacles"]),
    )
    config.validate()
    return config


class GridEnv(gym.Env[Coordinate, np.ndarray]):
    """Dynamic grid world with random start, goal, and obstacles each episode.

    Args:
        rows (int): Number of grid rows ``M``.
        cols (int): Number of grid columns ``N``.
        num_obstacles (int): Number of obstacles ``K`` to place per episode.

    Attributes:
        rows (int): Total number of rows in the grid layout.
        cols (int): Total number of columns in the grid layout.
        num_obstacles (int): Target number of obstacles enforced after reset.
        action_space (spaces.Discrete): Encodes the four cardinal moves.
        observation_space (spaces.Box): Binary layers for agent, goal, and obstacles.
        agent_position (Coordinate): Current agent coordinates in the grid.
        goal_position (Coordinate): Goal coordinates for the current episode.
        obstacle_positions (set[Coordinate]): Set of occupied obstacle coordinates.
        np_random (np.random.Generator): Environment RNG seeded through Gymnasium.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, rows: int, cols: int, num_obstacles: int) -> None:
        """Initializes the grid environment with validated dimensions and spaces.

        Args:
            rows (int): Number of grid rows ``M``.
            cols (int): Number of grid columns ``N``.
            num_obstacles (int): Number of obstacles ``K`` to place per episode.

        Returns:
            None
        """

        super().__init__()
        if rows < 2 or cols < 2:
            raise ValueError("rows and cols must both be >= 2")
        max_obstacles = rows * cols - 2
        if num_obstacles < 0 or num_obstacles > max_obstacles:
            raise ValueError(
                "num_obstacles must be between 0 and rows * cols - 2 "
                f"(received {num_obstacles}, limit {max_obstacles})"
            )

        self.rows = rows
        self.cols = cols
        self.num_obstacles = num_obstacles

        self.action_space = spaces.Discrete(len(ACTION_TO_DELTA))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, rows, cols),
            dtype=np.float32,
        )

        self.agent_position: Coordinate = (0, 0)
        self.goal_position: Coordinate = (rows - 1, cols - 1)
        self.obstacle_positions: set[Coordinate] = set()
        self.np_random: np.random.Generator | None = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Sequence[Coordinate]]] = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Resets the environment by sampling fresh start, goal, and obstacles.

        Args:
            seed (Optional[int]): Seed used to initialize the RNG for reproducibility.
            options (Optional[dict[str, Sequence[Coordinate]]]): Optional overrides for
                ``agent_position``, ``goal_position``, or ``obstacle_positions``. All
                supplied coordinates must lie inside the grid and be mutually unique.

        Returns:
            tuple[np.ndarray, dict[str, object]]: Observation composed of binary layers,
            and metadata containing the reset reason.
        """

        super().reset(seed=seed)
        assert self.np_random is not None  # set by super().reset
        overrides = options or {}

        if overrides:
            self._apply_overrides(overrides)
        else:
            self._sample_episode_layout()

        observation = self._build_observation()
        info = {"reason": "reset"}
        return observation, info

    def step(
        self, action: Action
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Executes a single action and reports the transition outcome.

        Args:
            action (Action): Discrete action index indicating the movement direction.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, object]]: Updated observation,
            reward signal, episode termination flag, truncation flag, and auxiliary info.
        """

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside the valid range 0-3")

        row_delta, col_delta = ACTION_TO_DELTA[action]
        candidate = (self.agent_position[0] + row_delta, self.agent_position[1] + col_delta)

        terminated = False
        truncated = False
        reward = -1.0
        reason = "moved"

        if not self._is_within_bounds(candidate):
            terminated = True
            reward = -10.0
            reason = "out_of_bounds"
        elif candidate in self.obstacle_positions:
            terminated = True
            reward = -10.0
            reason = "collision"
        elif candidate == self.goal_position:
            self.agent_position = candidate
            terminated = True
            reward = 10.0
            reason = "goal"
        else:
            self.agent_position = candidate

        observation = self._build_observation()
        info = {
            "reason": reason,
            "agent_position": self.agent_position,
            "goal_position": self.goal_position,
        }
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Displays the current grid state using ASCII characters.

        Args:
            mode (str): Either ``"human"`` to print to stdout or ``"ansi"`` to return a
                string representation.

        Returns:
            Optional[str]: Rendered grid when ``mode == "ansi"``; otherwise ``None``.
        """

        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for row, col in self.obstacle_positions:
            grid[row][col] = "#"
        goal_row, goal_col = self.goal_position
        grid[goal_row][goal_col] = "G"
        agent_row, agent_col = self.agent_position
        grid[agent_row][agent_col] = "A"

        output = "\n".join("".join(row) for row in grid)
        if mode == "human":
            print(output)
            return None
        if mode == "ansi":
            return output
        raise ValueError("Unsupported render mode. Use 'human' or 'ansi'.")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _apply_overrides(self, overrides: dict[str, Sequence[Coordinate]]) -> None:
        """Loads explicit layout coordinates provided via ``reset`` options.

        Args:
            overrides (dict[str, Sequence[Coordinate]]): Coordinate overrides for
                ``agent_position``, ``goal_position``, and/or ``obstacle_positions``.

        Returns:
            None
        """

        agent = overrides.get("agent_position")
        goal = overrides.get("goal_position")
        obstacles = overrides.get("obstacle_positions")

        if agent is None or goal is None or obstacles is None:
            raise ValueError(
                "overrides must include agent_position, goal_position, and "
                "obstacle_positions"
            )

        agent_coord = self._validate_coordinate(agent)
        goal_coord = self._validate_coordinate(goal)
        obstacle_coords = {self._validate_coordinate(obs) for obs in obstacles}

        if agent_coord == goal_coord:
            raise ValueError("Agent and goal positions must be distinct")
        if agent_coord in obstacle_coords or goal_coord in obstacle_coords:
            raise ValueError("Obstacles cannot overlap the agent or goal positions")
        if len(obstacle_coords) != self.num_obstacles:
            raise ValueError(
                f"Expected {self.num_obstacles} obstacles, received {len(obstacle_coords)}"
            )

        self.agent_position = agent_coord
        self.goal_position = goal_coord
        self.obstacle_positions = obstacle_coords

    def _sample_episode_layout(self) -> None:
        """Samples unique agent, goal, and obstacle coordinates for a new episode.

        Args:
            None

        Returns:
            None
        """

        assert self.np_random is not None
        total_cells = self.rows * self.cols
        required = self.num_obstacles + 2
        indices = self.np_random.permutation(total_cells)[:required]
        coordinates = [divmod(index, self.cols) for index in indices]

        self.agent_position = coordinates[0]
        self.goal_position = coordinates[1]
        self.obstacle_positions = set(coordinates[2:])

    def _build_observation(self) -> np.ndarray:
        """Constructs the layered observation tensor describing the grid state.

        Args:
            None

        Returns:
            np.ndarray: Binary tensor with agent, goal, and obstacle layers.
        """

        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        agent_row, agent_col = self.agent_position
        goal_row, goal_col = self.goal_position
        observation[0, agent_row, agent_col] = 1.0
        observation[1, goal_row, goal_col] = 1.0
        for row, col in self.obstacle_positions:
            observation[2, row, col] = 1.0
        return observation

    def _is_within_bounds(self, coordinate: Coordinate) -> bool:
        """Determines whether a coordinate lies inside the configured grid.

        Args:
            coordinate (Coordinate): Row and column indices being validated.

        Returns:
            bool: ``True`` when the coordinate is inside the grid bounds.
        """

        row, col = coordinate
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _validate_coordinate(self, value: Sequence[int]) -> Coordinate:
        """Normalizes a coordinate sequence and ensures it is valid for the grid.

        Args:
            value (Sequence[int]): Candidate row/column pair to validate.

        Returns:
            Coordinate: Tuple containing the sanitized coordinate values.
        """

        if len(value) != 2:
            raise ValueError("Coordinates must contain exactly two integers")
        row, col = int(value[0]), int(value[1])
        if not self._is_within_bounds((row, col)):
            raise ValueError(f"Coordinate {(row, col)!r} lies outside the grid bounds")
        return row, col


def register_grid_env(env_id: str = "RLAssignment/GridEnv-v0") -> None:
    """Registers ``GridEnv`` with Gymnasium to enable ``gym.make`` usage.

    Args:
        env_id (str): Identifier used for environment registration.

    Returns:
        None: The registration has the side effect of updating Gymnasium's registry.
    """

    from gymnasium.envs.registration import register
    from gymnasium.error import Error

    try:
        register(
            id=env_id,
            entry_point="environment:GridEnv",
        )
    except Error:
        # Registration already exists; ignore to keep helper idempotent.
        pass


def create_env_from_config(config: GridConfig) -> GridEnv:
    """Instantiates ``GridEnv`` based on validated ``GridConfig`` values.

    Args:
        config (GridConfig): Validated configuration describing the grid setup.

    Returns:
        GridEnv: New environment instance parameterized by the provided config.
    """

    return GridEnv(config.grid_rows, config.grid_cols, config.num_obstacles)
