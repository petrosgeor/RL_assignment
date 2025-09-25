"""Grid-based Gymnasium environment for Part 2 of the RL assignment."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from collections import deque
import numpy as np


# Potential-shaping coefficients (used when combined rewards are enabled)
ALPHA_POT: float = 0.4   # scale for geodesic shaping
GAMMA_POT: float = 1     # potential discount (1 = pure change)

Action = int
Coordinate = tuple[int, int]

# Mapping from discrete action index to row/column deltas.
ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


@dataclass(frozen=True)
class GridConfig:
    """Stores the grid dimensions and obstacle count for the environment.

    Args:
        grid_rows (int): Number of rows M in the grid.
        grid_cols (int): Number of columns N in the grid.
        num_obstacles (int): Number of static obstacles K to place each episode.

    Attributes:
        grid_rows (int): Stored number of grid rows.
        grid_cols (int): Stored number of grid columns.
        num_obstacles (int): Stored number of obstacles enforced at reset.
    """

    grid_rows: int
    grid_cols: int
    num_obstacles: int

    def validate(self) -> None:
        if self.grid_rows < 2 or self.grid_cols < 2:
            raise ValueError("grid_rows and grid_cols must both be >= 2")
        max_obstacles = self.grid_rows * self.grid_cols - 2
        if self.num_obstacles < 0 or self.num_obstacles > max_obstacles:
            msg = (
                "num_obstacles must be between 0 and grid_rows * grid_cols - 2 "
                f"(received {self.num_obstacles}, limit {max_obstacles})"
            )
            raise ValueError(msg)


def load_grid_config(path= "training_configurations/env_config.yaml",) -> GridConfig:
    """Loads and validates the grid configuration stored in YAML format.
    Args:
        path: Location of the YAML file describing the grid. By
            default, this reads from training_configurations/env_config.yaml.

    Returns:
        GridConfig: Validated configuration ready for environment construction.
    """

    cfg_path = Path(path)
    # Read YAML configuration from disk
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    # Construct dataclass and validate bounds/constraints
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
        rows (int): Number of grid rows M.
        cols (int): Number of grid columns N.
        num_obstacles (int): Number of obstacles K to place per episode.

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
            rows (int): Number of grid rows M.
            cols (int): Number of grid columns N.
            num_obstacles (int): Number of obstacles K to place per episode.

        Returns:
            None
        """

        super().__init__()
        # Validate dimensions and obstacle budget
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

        # Discrete 4-action space (up, down, left, right)
        self.action_space = spaces.Discrete(len(ACTION_TO_DELTA))
        # Observation has 3 binary layers: agent, goal, obstacles
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
        # Geodesic distance map cache (computed at reset)
        self._dist_map = None  # filled at reset when shaping is enabled
        self.use_combined_rewards = False

    def set_use_combined_rewards(self, flag: bool) -> None:
        """Enable or disable potential-based shaping."""

        self.use_combined_rewards = bool(flag)

    def reset(
        self,
        seed = None,
        options = None,
    ) -> tuple:
        """Resets the environment by sampling fresh a start, goal, and obstacles.

        Args:
            seed: Seed used to initialize the RNG for reproducibility.
            options : Optional dictionary overriding the
                agent_position, goal_position, or obstacle_positions. All
                supplied coordinates must lie inside the grid and be mutually unique.

        Returns:
            tuple[np.ndarray, dict[str, object]]: Observation composed of binary layers,
            and metadata containing the reset reason.
        """

        super().reset(seed=seed)
        assert self.np_random is not None  # set by super().reset
        overrides = options or {}

        # Either apply explicit layout or sample a fresh one
        if overrides:
            self._apply_overrides(overrides)
        else:
            self._sample_episode_layout()

        # Build geodesic map once per episode (used when shaping is enabled)
        self._dist_map = self._compute_geodesic_distance_map()

        observation = self._build_observation()
        info = {"reason": "reset"}
        return observation, info

    def step(self, action: Action) -> tuple:
        """Executes a single action and reports the transition outcome.

        Args:
            action (Action): Discrete action index indicating the movement direction.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, object]]: Updated observation,
            reward signal, episode termination flag, truncation flag, and auxiliary info.
        """

        # Validate action index
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside the valid range 0-3")

        # Translate discrete action to row/col delta
        row_delta, col_delta = ACTION_TO_DELTA[action]  # Map the action
        curr = self.agent_position
        candidate = (curr[0] + row_delta, curr[1] + col_delta)  # next position

        # Initialize termination flags and base reward
        # (base reward is step-penalty unless terminal events occur)
        terminated = False
        truncated = False
        reward = -1.0
        reason = "moved"

        # Resolve transition outcome in priority order
        if not self._is_within_bounds(candidate):
            terminated = True
            reward = -10.0
            reason = "out_of_bounds"
        elif candidate in self.obstacle_positions:
            terminated = True
            reward = -10.0
            reason = "collision"
        elif candidate == self.goal_position:
            # Goal reached: move agent and end episode with goal reward
            self.agent_position = candidate
            terminated = True
            reward = 10.0
            reason = "goal"
        else:
            # Regular move: update position only
            self.agent_position = candidate

        # Add potential-based shaping only if enabled via config/flag
        if getattr(self, "use_combined_rewards", False):
            reward += self._potential_bonus_geodesic(curr, candidate, reason)

        # Build next observation and attach minimal info for diagnostics
        observation = self._build_observation()
        info = {
            "reason": reason,
            "agent_position": self.agent_position,
            "goal_position": self.goal_position,
        }
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> str:
        """Displays the current grid state using ASCII characters.

        Args:
            mode (str): Either human to print to stdout or ansi to return a
                string representation.

        Returns:
            str: Rendered grid when mode == ansi; otherwise None.
        """

        # Create an ASCII grid and mark obstacles, goal, and agent
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for row, col in self.obstacle_positions:
            grid[row][col] = "#"
        goal_row, goal_col = self.goal_position
        grid[goal_row][goal_col] = "G"
        agent_row, agent_col = self.agent_position
        grid[agent_row][agent_col] = "A"

        output = "\n".join("".join(row) for row in grid)
        if mode == "human":
            # Print to stdout for an immediate visualisation
            print(output)
            return None
        if mode == "ansi":
            # Return string for programmatic or file-based consumption
            return output
        raise ValueError("Unsupported render mode. Use 'human' or 'ansi'.")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _apply_overrides(self, overrides: dict) -> None:
        """Loads explicit layout coordinates provided via reset options.

        Args:
            overrides (dict): Coordinate overrides for
                agent_position, goal_position, and/or obstacle_positions.

        Returns:
            None

        Example:
            An overrides mapping must include all three entries and match the configured number of obstacles.
            Concretely we might have: 
            agent_position: (0, 0)
            goal_position: (5, 5)
            obstacle_positions: [(1, 1), (2, 3), (4, 4)]
        """

        # Pull expected keys from options
        agent = overrides.get("agent_position")
        goal = overrides.get("goal_position")
        obstacles = overrides.get("obstacle_positions")

        if agent is None or goal is None or obstacles is None:
            raise ValueError(
                "overrides must include agent_position, goal_position, and obstacle_positions"
            )

        # Validate and normalize all coordinates
        agent_coord = self._validate_coordinate(agent)
        goal_coord = self._validate_coordinate(goal)
        obstacle_coords = {self._validate_coordinate(obs) for obs in obstacles}

        # Enforce mutual exclusivity and obstacle count
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

        assert self.np_random is not None  # ensure RNG is ready (set in reset)
        total_cells = self.rows * self.cols  # total number of cells in the grid
        required = self.num_obstacles + 2  # sample agent + goal + all obstacles
        # Draw a random permutation and take the first 'required' unique cells
        indices = self.np_random.permutation(total_cells)[:required]  # unique cells
        # Convert flat cell index k into (row, col) via divmod(k, cols)
        coordinates = [divmod(index, self.cols) for index in indices]  # to (row, col)

        self.agent_position = coordinates[0]  # first cell becomes agent start
        self.goal_position = coordinates[1]  # second cell becomes goal position
        self.obstacle_positions = set(coordinates[2:])  # remaining cells are obstacles

    def _build_observation(self) -> np.ndarray:
        """Constructs the layered observation tensor describing the grid state.

        Args:
            None

        Returns:
            np.ndarray: 3D Binary tensor with agent, goal, and obstacle layers.

        The first layer shows where the agent is. The second the goal position and the third one the obstacle positions 
        """

        # Start with zeroed layers and set ones at active positions
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        agent_row, agent_col = self.agent_position
        goal_row, goal_col = self.goal_position
        observation[0, agent_row, agent_col] = 1.0
        observation[1, goal_row, goal_col] = 1.0
        for row, col in self.obstacle_positions:
            observation[2, row, col] = 1.0
        return observation

    def _compute_geodesic_distance_map(self) -> np.ndarray:
        """Compute obstacle-aware shortest-path distances to the goal via BFS.

        Returns an int32 array dist[r, c] with the minimum number of 4-connected
        steps from (r, c) to the goal, respecting obstacles. Unreachable cells
        receive D_MAX = rows + cols.
        """

        rows, cols = self.rows, self.cols
        D_MAX = rows * cols + 1
        # Initialize all distances to D_MAX (mark unreachable by default)
        dist = np.full((rows, cols), D_MAX, dtype=np.int32)

        # Precompute obstacle mask for fast checks
        blocked = np.zeros((rows, cols), dtype=bool)
        for (r, c) in self.obstacle_positions:
            blocked[r, c] = True

        gr, gc = self.goal_position
        if not (0 <= gr < rows and 0 <= gc < cols) or blocked[gr, gc]:
            # If goal is invalid or blocked, no cell can reach it
            return dist

        # BFS from the goal to compute 4-connected shortest-path distances
        dist[gr, gc] = 0
        q = deque([(gr, gc)])
        while q:
            r, c = q.popleft()
            d = dist[r, c] + 1
            for dr, dc in ACTION_TO_DELTA.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not blocked[nr, nc]:
                    if d < dist[nr, nc]:
                        dist[nr, nc] = d
                        q.append((nr, nc))
        return dist

    def _potential_bonus_geodesic(self, curr: Coordinate, nxt: Coordinate, reason: str) -> float:
        """Concave geodesic shaping using phi(d)=log(1+d).

        Returns ALPHA_POT * (phi(d_curr) - GAMMA_POT * phi(d_next)) where d are
        BFS geodesic distances to the goal. Shaping is suppressed on invalid
        terminations and when distances are unreachable.
        """

        # Do not shape invalid terminations
        if reason in ("collision", "out_of_bounds"):
            return 0.0
        # If no distance map is available, skip shaping
        if self._dist_map is None:
            return 0.0
        D_MAX = self.rows * self.cols + 1
        d_curr = int(self._dist_map[curr[0], curr[1]])
        d_next = int(self._dist_map[nxt[0], nxt[1]])
        # Skip cells that are unreachable from the goal
        if d_curr >= D_MAX or d_next >= D_MAX:
            return 0.0
        phi_curr = float(np.log1p(d_curr))
        phi_next = float(np.log1p(d_next))
        return ALPHA_POT * (phi_curr - GAMMA_POT * phi_next)

    def _is_within_bounds(self, coordinate: Coordinate) -> bool:
        """Determines whether a coordinate lies inside the configured grid.

        Args:
            coordinate (Coordinate): Row and column indices being validated.

        Returns:
            bool: True when the coordinate is inside the grid bounds.
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

        # Enforce two components and cast to ints
        if len(value) != 2:
            raise ValueError("Coordinates must contain exactly two integers")
        row, col = int(value[0]), int(value[1])
        if not self._is_within_bounds((row, col)):
            raise ValueError(f"Coordinate {(row, col)!r} lies outside the grid bounds")
        return row, col




def create_env_from_config(config: GridConfig) -> GridEnv:
    """Instantiates GridEnv based on validated GridConfig values.

    Args:
        config (GridConfig): Validated configuration describing the grid setup.

    Returns:
        GridEnv: New environment instance parameterized by the provided config.
    """

    env = GridEnv(config.grid_rows, config.grid_cols, config.num_obstacles)
    return env
