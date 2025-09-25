import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import yaml
from stable_baselines3 import DQN
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils import build_env, print_summary, save_model

from training.training_evaluation import evaluate_model, online_eval_and_log
from training.early_stopping import EarlyStopping


def load_dqn_config(path: str) -> dict:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Configuration file {cfg_path} is empty")
    return raw



def build_model(cfg: dict, env) -> DQN:
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    return DQN(
        policy=cfg["policy"],
        env=env,
        learning_rate=float(cfg["learning_rate"]),
        buffer_size=int(cfg["buffer_size"]),
        learning_starts=int(cfg["learning_starts"]),
        batch_size=int(cfg["batch_size"]),
        gamma=float(cfg["gamma"]),
        train_freq=int(cfg["train_freq"]),
        gradient_steps=int(cfg["gradient_steps"]),
        target_update_interval=int(cfg["target_update_interval"]),
        exploration_fraction=float(cfg["exploration_fraction"]),
        exploration_final_eps=float(cfg["exploration_final_eps"]),
        max_grad_norm=float(cfg["max_grad_norm"]),
        seed=int(cfg["seed"]),
        device=device,
        tensorboard_log=None,
        policy_kwargs=cfg.get("policy_kwargs") or None,
        verbose=0,
    )


def train_model(model: DQN, cfg: dict, best_path: Path) -> int:
    total_timesteps = int(cfg["total_timesteps"])
    eval_every_ts = int(cfg.get("eval_every_timesteps", 0) or 0)
    chunk = eval_every_ts if eval_every_ts > 0 else 5000

    trained = 0
    iterations = 0
    next_eval_ts = eval_every_ts if eval_every_ts > 0 else None

    if eval_every_ts > 0:
        stopper = EarlyStopping(
            patience=10,
            min_delta=0.02,
            verbose=True,
            save_best=True,
            best_path=best_path,
        )
    else:
        stopper = None
    if stopper is None:
        print("[early-stop] Disabled (eval_every_timesteps=0)")

    while trained < total_timesteps:
        remain = total_timesteps - trained  # remaining training steps
        this_chunk = min(chunk, remain)
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=(trained == 0))

        trained += this_chunk
        iterations += 1
        print(f"[dqn] iter={iterations} ts={trained}/{total_timesteps}")

        if next_eval_ts is not None and trained >= next_eval_ts:
            success = online_eval_and_log(model, cfg, trained_ts=trained, iter_idx=iterations) # repressents the success rate (it's in [0,1]), and is used by the stopper class
            if stopper and stopper(success, model):
                best = stopper.best()
                best_str = f"{best:.2%}" if best is not None else "n/a"
                print(f"[early-stop] Triggered at iter={iterations} ts={trained} (best success={best_str})")
                break
            next_eval_ts += eval_every_ts

    return iterations


def main() -> None:

    parser = argparse.ArgumentParser(description="Train DQN on GridEnv")

    parser.add_argument(
        "--reward",
        choices=("base", "combined"),
        default="base",
        help="Reward type: base or combined (with potential shaping)",
    )
    args = parser.parse_args()

    cfg_path = REPO_ROOT / "training_configurations" / "dqn_config.yaml"

    cfg = load_dqn_config(cfg_path)
    cfg["reward_mode"] = args.reward
    print_summary(SimpleNamespace(**cfg))

    use_combined = args.reward == "combined"
    save_path = REPO_ROOT / "saved_models" / f"dqn_model_{args.reward}.zip"
    cfg["save_path"] = str(save_path)

    if not cfg.get("train_then_eval", True):
        print("train_then_eval disabled; exiting without training.")
        return

    env = build_env(SimpleNamespace(**cfg), vec=False)
    env.unwrapped.set_use_combined_rewards(use_combined)
    try:
        model = build_model(cfg, env)
        iterations = train_model(model, cfg, save_path)
    finally:
        env.close()

    save_model(model, save_path)

    # Final evaluation and summary printout
    metrics = evaluate_model(model, cfg)
    print(
        "Evaluation complete | episodes: {episodes:.0f} | avg return: {ret:.2f} | "
        "avg length: {length:.2f} | success rate: {succ:.2%}".format(
            episodes=metrics["episodes"],
            ret=metrics["average_return"],
            length=metrics["average_length"],
            succ=metrics["success_rate"],
        )
    )



if __name__ == "__main__":
    main()
