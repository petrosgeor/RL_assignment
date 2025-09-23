from types import SimpleNamespace
import numpy as np

from training.utils import build_env, to_scalar_action


def evaluate_model(model, cfg: dict) -> dict:
    """Runs evaluation episodes and computes summary metrics."""

    episodes = int(cfg.get("eval_episodes", 100))
    env = build_env(SimpleNamespace(**cfg), vec=False)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

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


def online_eval_and_log(model, cfg: dict, trained_ts: int, iter_idx: int) -> float:
    """Runs a lightweight evaluation during training and returns the success rate."""

    episodes = max(int(cfg.get("online_eval_episodes", 10)), 1)
    eval_cfg = dict(cfg)
    eval_cfg["eval_episodes"] = episodes
    metrics = evaluate_model(model, eval_cfg)

    print(
        "[eval] ts={ts} iter={it} episodes={eps:.0f} avg_return={ret:.2f} "
        "avg_length={length:.2f} success_rate={succ:.2%}".format(
            ts=int(trained_ts),
            it=int(iter_idx),
            eps=metrics["episodes"],
            ret=metrics["average_return"],
            length=metrics["average_length"],
            succ=metrics["success_rate"],
        )
    )

    return float(metrics["success_rate"])
