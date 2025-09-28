"""Value iteration with unified absorbing terminals on a padded canvas.

We represent a fixed grid layout using a 3 * (H) * (W) observation with
H = M + 2 and W = N + 2. The interior cells 1..M * 1..N are the true grid; the
outer border (rows 0 and M+1, cols 0 and N+1) is marked as obstacles to model OOB cells uniformly.

Layers:
- 0: agent one hot
- 1: goal one hot
- 2: obstacles (including the full outer border)

Absorbing terminals:
- Entering the goal pays +10 on that transition; afterwards p(s, 0 | s, a) = 1.
- Entering an obstacle (including border/OOB) pays −10; afterwards p(s, 0 | s, a) = 1.

Regular moves pay −1. The one‑step MDP is deterministic, so p(s', r | s, a) is a point mass.
"""

from typing import Iterable, Tuple
import numpy as np
from pathlib import Path
import sys 
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from environment import load_grid_config, create_env_from_config

ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


def build_state(
    M: int,
    N: int,
    agent: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Iterable[Tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build a 3 × (M+2) × (N+2) observation with border obstacles.

    Args:
        M (int): Interior rows.
        N (int): Interior cols.
        agent (tuple[int,int]): Interior coordinates (0..M-1, 0..N-1).
        goal (tuple[int,int]): Interior coordinates (0..M-1, 0..N-1).
        obstacles (Iterable[tuple[int,int]] | None): Interior obstacle coords.

    Returns:
        np.ndarray: Observation tensor shaped (3, M+2, N+2).
    """
    H, W = int(M) + 2, int(N) + 2
    obs = np.zeros((3, H, W), dtype=np.float32)

    # Border obstacles to represent OOB
    obs[2, 0, :] = 1.0
    obs[2, H - 1, :] = 1.0
    obs[2, :, 0] = 1.0
    obs[2, :, W - 1] = 1.0

    # Interior obstacles (shift by +1)
    if obstacles is not None:
        for r, c in obstacles:
            rr, cc = int(r) + 1, int(c) + 1
            obs[2, rr, cc] = 1.0

    # Place goal and agent (shift by +1)
    gr, gc = int(goal[0]) + 1, int(goal[1]) + 1
    ar, ac = int(agent[0]) + 1, int(agent[1]) + 1

    # Validate positions
    if obs[2, gr, gc] > 0.5:
        raise ValueError("Goal cannot overlap an obstacle or border")
    # Agent may also start on an obstacle/border to represent an absorbing terminal.
    # Agent may coincide with goal to represent a terminal state.

    obs[1, gr, gc] = 1.0
    obs[0, ar, ac] = 1.0
    return obs


def _is_terminal(state: np.ndarray) -> str | None:
    """Returns 'goal' or 'obstacle' when agent stands on that cell; otherwise it returns None, indicating a normal cell.

    Args:
        state (np.ndarray): Observation tensor (3, H, W) on the padded canvas where H = M+2 and W = N+2

    Returns:
        str | None: Terminal state or None if non-terminal state.
    """

    idx = np.argwhere(state[0] > 0.5)
    assert idx.shape[0] == 1, "State must contain exactly one agent marker"
    r, c = int(idx[0][0]), int(idx[0][1])
    on_goal = state[1, r, c] > 0.5
    on_obs = state[2, r, c] > 0.5
    assert not (on_goal and on_obs), "Agent cannot be on goal and obstacle simultaneously"
    if on_goal:
        return "goal"
    if on_obs:
        return "obstacle"
    return None


def compute_transition_probability(
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    reward: float,
) -> float:
    """Return p(s', r | s, a) for the deterministic, padded‑canvas MDP.

    From terminals, we self‑loop with reward 0. Otherwise we compute the unique
    next agent cell by adding the action delta and pick reward by the content of that cell: +10 for goal, −10 for obstacle/border, −1 for free.
    """

    if action not in ACTION_TO_DELTA:
        return 0.0
    if state.shape != next_state.shape or state.ndim != 3 or state.shape[0] != 3:
        return 0.0

    if _is_terminal(state) is not None:
        return 1.0 if (np.array_equal(next_state, state) and float(reward) == 0.0) else 0.0

    ar, ac = map(int, np.argwhere(state[0] > 0.5)[0])
    dr, dc = ACTION_TO_DELTA[action]
    nr, nc = ar + dr, ac + dc

    if state[1, nr, nc] > 0.5:
        actual_reward = 10.0
    elif state[2, nr, nc] > 0.5:
        actual_reward = -10.0
    else:
        actual_reward = -1.0

    det_next = np.zeros_like(state, dtype=np.float32)
    det_next[0, nr, nc] = 1.0
    det_next[1] = state[1]
    det_next[2] = state[2]
    # Only return 1 if the next candidate position of the agent (passed as argument) equals the deterministic next state and the reward equals the deterministic reward. Otherwise return 0
    return 1.0 if (np.array_equal(next_state, det_next) and abs(float(reward) - actual_reward) <= 1e-9) else 0.0


def find_expectation(state: np.ndarray, action: int, V: np.ndarray, gamma: float = 0.99) -> float:
    """Return E[r + gamma V_k(s') | s, a] for deterministic transitions.

    V is a 2D value grid defined over the padded canvas.
    """

    # Absorbing: no continuation, value at terminals is 0 
    if _is_terminal(state) is not None:
        return 0.0

    ar, ac = map(int, np.argwhere(state[0] > 0.5)[0])
    dr, dc = ACTION_TO_DELTA.get(action, (0, 0))
    nr, nc = ar + dr, ac + dc

    if state[1, nr, nc] > 0.5:
        r = 10.0
    elif state[2, nr, nc] > 0.5:
        r = -10.0
    else:
        r = -1.0

    return float(r + gamma * V[nr, nc])


def value_iteration(
    M: int,
    N: int,
    goal: Tuple,
    obstacles: Tuple = None,
    gamma: float = 0.99,
    theta: float = 1e-4,
    max_iters: int = 1000,
) -> np.ndarray:
    """Run value iteration on a fixed layout; return optimal V as (H, W) grid.

    H = M + 2, W = N + 2. Border/obstacle cells are absorbing with value 0.
    """

    H, W = int(M) + 2, int(N) + 2
    actions = (0, 1, 2, 3)
    V = np.zeros((H, W), dtype=np.float32)

    for _ in range(int(max_iters)):
        delta = 0.0
        V_next = V.copy()
        for r in range(1, H - 1):
            for c in range(1, W - 1):   # Iterate over all states s
                s = build_state(M, N, (r - 1, c - 1), goal, obstacles)
                if _is_terminal(s) is not None:
                    V_next[r, c] = 0.0
                    continue
                best = max(find_expectation(s, a, V, gamma) for a in actions)
                delta = max(delta, abs(best - float(V[r, c])))
                V_next[r, c] = best
        V = V_next
        if delta < float(theta):
            break

    return V



def pick_action(V_star, agent_rc, gamma, M, N, goal_rc, obstacles_rc):
    """
    Selects the greedy action under V* using one-step lookahead:
        argmax_a E[r + gamma * V*(s') | s, a]
    where the expectation is computed by the existing `find_expectation` helper.
    Args:
        V_star: 2D numpy array of shape (H, W) with optimal state values on the padded canvas.
        agent_rc: (row, col) in INTERIOR coords (0..M-1, 0..N-1) for the current agent position.
        gamma: discount factor in (0, 1].
        M, N: interior grid size used to construct the padded canvas (H=M+2, W=N+2).
        goal_rc: (row, col) in INTERIOR coords for the goal.
        obstacles_rc: iterable of (row, col) INTERIOR coords for obstacles.
    Returns:
        int in {0,1,2,3} (0=up, 1=down, 2=left, 3=right).
    """
    # Build the padded-canvas observation for the current agent position
    state = build_state(M, N, agent_rc, goal_rc, obstacles_rc)

    # If we're in an absorbing state, pick any action uniformly at random
    if _is_terminal(state) is not None:
        return int(np.random.default_rng().integers(0, 4))

    actions = (0, 1, 2, 3)  # up, down, left, right
    best_action = 0
    best_score = -float("inf")

    # One-step lookahead using the deterministic model via find_expectation
    for a in actions:
        score = find_expectation(state, a, V_star, gamma)
        if score > best_score:
            best_score = score
            best_action = a

    return int(best_action)




if __name__ == "__main__":
    # Load a random layout from the base environment, then run value iteration on it.
    

    cfg = load_grid_config(REPO_ROOT / "training_configurations" / "env_config.yaml")
    env = create_env_from_config(cfg)

    # Evaluate over 100 episodes: each episode samples a fresh layout,
    # runs value iteration to obtain V*, then rolls out greedily.
    episodes = 100
    gamma = 0.99
    successes = 0
    lengths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        rows, cols = obs.shape[1], obs.shape[2]
        goal_rc = tuple(map(int, np.argwhere(obs[1] > 0.5)[0]))
        obstacles_rc = [tuple(map(int, xy)) for xy in np.argwhere(obs[2] > 0.5)]

        V_star = value_iteration(rows, cols, goal_rc, obstacles_rc, gamma=gamma, theta=1e-5, max_iters=500)

        terminated = False
        truncated = False
        info = {}
        obs_t = obs
        ep_len = 0
        while not (terminated or truncated):
            agent_rc = tuple(map(int, np.argwhere(obs_t[0] > 0.5)[0]))
            a = pick_action(V_star, agent_rc, gamma, rows, cols, goal_rc, obstacles_rc)
            obs_t, reward, terminated, truncated, info = env.step(int(a))
            ep_len += 1
        if info.get("reason") == "goal":
            successes += 1
        lengths.append(ep_len)

    rate = successes / episodes
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    print(f"Success rate over {episodes} episodes: {successes}/{episodes} = {rate:.2%}")
    print(f"Average trajectory length over {episodes} episodes: {avg_len:.2f}")
