import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import GridConfig, load_grid_config

Action = int
Coordinate = tuple[int, int]

ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


NUM_BONUS = 1      # number of bonus cells to collect per episode
C_STEP = 0.1       # per-step penalty (applies to normal moves)
R_BONUS = 4.0      # reward when first visiting a bonus cell
R_GOAL = 100.0      # terminal reward when all bonuses collected and goal reached
P_PREM = 5.0       # penalty for reaching goal before collecting all bonuses
C_FAIL = 10.0      # penalty for collision or out-of-bounds


class ExpandedGridEnv(gym.Env):
    """Grid with obstacles, goal, and first‑visit bonus positions.

    Args:
        rows (int): Number of grid rows M.
        cols (int): Number of grid columns N.
        num_obstacles (int): Number of obstacles K to place per episode.

    Attributes:
        rows (int): Total number of rows in the grid.
        cols (int): Total number of columns in the grid.
        num_obstacles (int): Obstacles enforced after reset.
        action_space (spaces.Discrete): Encodes four cardinal actions.
        observation_space (spaces.Box): Four layers: agent, goal, obstacles, remaining bonuses.
        agent_position (Coordinate): Current agent (row, col).
        goal_position (Coordinate): Goal (row, col).
        obstacle_positions (set[Coordinate]): Set of blocked cells.
        bonus_positions (set[Coordinate]): Fixed bonus cells for the episode.
        visited_bonus (set[Coordinate]): Subset of bonus_positions visited so far.
        np_random (np.random.Generator | None): RNG seeded by Gymnasium.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, rows: int, cols: int, num_obstacles: int) -> None:
        """Initializes a validated grid and observation/action spaces.

        Args:
            rows (int): Grid rows.
            cols (int): Grid cols.
            num_obstacles (int): Obstacles count per episode.

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
            shape=(4, rows, cols),  # agent, goal, obstacles, remaining bonuses
            dtype=np.float32,
        )

        self.agent_position = (0, 0)
        self.goal_position = (rows - 1, cols - 1)
        self.obstacle_positions = set()
        self.bonus_positions = set()
        self.visited_bonus = set()
        self.np_random = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None) -> tuple:
        """Sample a fresh episode layout (including bonuses) and reset state."""

        super().reset(seed=seed)
        assert self.np_random is not None  # set by super().reset
        overrides = options or {}
        # If caller provided explicit coordinates in options, apply them; otherwise sample

        if overrides:
            self._apply_overrides(overrides)
        else:
            self._sample_episode_layout()

        # Clear first-visit bookkeeping for a new episode
        self.visited_bonus = set()
        observation = self._build_observation()
        info = {"reason": "reset", "bonuses_remaining": NUM_BONUS}
        return observation, info

    def step(self, action: Action) -> tuple:
        """Apply one action; add first‑visit bonus on new bonus cells."""

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside the valid range 0-3")

        # Translate action into grid motion
        row_delta, col_delta = ACTION_TO_DELTA[action]
        candidate = (self.agent_position[0] + row_delta, self.agent_position[1] + col_delta)

        terminated = False
        truncated = False
        reward = -C_STEP
        reason = "moved"

        # Out-of-bounds or collision (terminal)
        if not self._is_within_bounds(candidate):
            terminated = True
            reward = -C_FAIL
            reason = "out_of_bounds"
        # Collided with obstacle
        elif candidate in self.obstacle_positions:
            terminated = True
            reward = -C_FAIL
            reason = "collision"
        # Reached the goal position
        elif candidate == self.goal_position:
            # Goal reached; pay only after collecting all bonuses
            terminated = True
            # Give the final goal reward (positive) if all bonus positions where visited
            if len(self.visited_bonus) == NUM_BONUS:
                reward = R_GOAL
                reason = "goal"
            # If at least one bonus position is not visited, then we penalize the agent.
            else:
                reward = -P_PREM
                reason = "premature_goal"
        else:
            # Regular move: update position and pay first-visit bonus if any
            self.agent_position = candidate
            if candidate in self.bonus_positions and candidate not in self.visited_bonus:
                self.visited_bonus.add(candidate)
                reward += R_BONUS

        observation = self._build_observation()
        info = {
            "reason": reason,
            "visited": len(self.visited_bonus),
            "remaining": NUM_BONUS - len(self.visited_bonus),
        }
        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_overrides(self, overrides: dict) -> None:
        """Load explicit coordinates provided via reset options."""

        agent = overrides.get("agent_position")
        goal = overrides.get("goal_position")
        obstacles = overrides.get("obstacle_positions")
        bonuses = overrides.get("bonus_positions")

        if agent is None or goal is None or obstacles is None:
            raise ValueError(
                "overrides must include agent_position, goal_position, and obstacle_positions"
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

        # Handle bonuses: optional override or random placement later
        if bonuses is not None:
            bonus_coords = {self._validate_coordinate(b) for b in bonuses}
            if len(bonus_coords) != NUM_BONUS:
                raise ValueError(f"Expected {NUM_BONUS} bonuses, received {len(bonus_coords)}")
            if (
                self.agent_position in bonus_coords
                or self.goal_position in bonus_coords
                or any(b in self.obstacle_positions for b in bonus_coords)
            ):
                raise ValueError("Bonus cells must be distinct from agent, goal, and obstacles")
            self.bonus_positions = bonus_coords
        else:
            # If not provided, sample later to avoid duplicating logic here
            self._sample_bonuses()

    def _sample_episode_layout(self) -> None:
        """Sample unique agent, goal, obstacles, and bonus coordinates."""

        assert self.np_random is not None
        total_cells = self.rows * self.cols
        # We need positions for: agent, goal, obstacles, and bonuses
        required = self.num_obstacles + 2 + NUM_BONUS
        # Sample distinct flat indices in [0, rows*cols) then take the first required
        indices = self.np_random.permutation(total_cells)[:required]
        # Map each flat index k to 2D coordinates (k // cols, k % cols)
        coordinates = [divmod(int(idx), self.cols) for idx in indices]

        # Assign sampled positions by convention: [agent, goal, obstacles..., bonuses...]
        self.agent_position = coordinates[0]
        self.goal_position = coordinates[1]
        self.obstacle_positions = set(coordinates[2 : 2 + self.num_obstacles])
        self.bonus_positions = set(coordinates[2 + self.num_obstacles :])

    def _sample_bonuses(self) -> None:
        """Sample NUM_BONUS cells avoiding agent, goal, and obstacles."""

        assert self.np_random is not None
        # Exclude existing occupied cells so bonuses don't overlap agent/goal/obstacles
        forbidden = {self.agent_position, self.goal_position} | self.obstacle_positions
        # Enumerate all grid cells and filter forbidden ones
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        candidates = [cell for cell in all_cells if cell not in forbidden]
        # Randomly permute candidate indices and pick the first NUM_BONUS
        perm = self.np_random.permutation(len(candidates))
        chosen = [candidates[int(i)] for i in perm[:NUM_BONUS]]
        self.bonus_positions = set(chosen)

    def _build_observation(self) -> np.ndarray:
        """Build a 4‑layer observation: agent, goal, obstacles, remaining bonuses."""

        # Start with all‑zero layers and mark active cells as 1.0
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        a_row, a_col = self.agent_position
        g_row, g_col = self.goal_position
        observation[0, a_row, a_col] = 1.0
        observation[1, g_row, g_col] = 1.0
        for row, col in self.obstacle_positions:
            observation[2, row, col] = 1.0
        # Remaining bonuses only (visited disappear from layer 3 and become 0)
        for row, col in (self.bonus_positions - self.visited_bonus):
            observation[3, row, col] = 1.0
        return observation

    def _is_within_bounds(self, coordinate: Coordinate) -> bool:
        """Return True if coordinate lies inside the grid bounds."""

        row, col = coordinate
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _validate_coordinate(self, value) -> Coordinate:
        """Normalize a sequence into a valid grid coordinate."""

        if len(value) != 2:
            raise ValueError("Coordinates must contain exactly two integers")
        row, col = int(value[0]), int(value[1])
        if not self._is_within_bounds((row, col)):
            raise ValueError(f"Coordinate {(row, col)!r} lies outside the grid bounds")
        return row, col


def create_env_from_config(config: GridConfig) -> ExpandedGridEnv:
    """Instantiates ExpandedGridEnv using a validated GridConfig.

    Args:
        config (GridConfig): Grid dimensions and obstacle count.

    Returns:
        ExpandedGridEnv: New environment instance parameterized by config.
    """

    return ExpandedGridEnv(config.grid_rows, config.grid_cols, config.num_obstacles)
