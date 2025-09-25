import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.utils import build_env, print_summary, save_model
from training.training_evaluation import online_eval_and_log
from training.early_stopping import EarlyStopping


def load_ppo_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Configuration file {path} is empty")
    return raw


def build_model(cfg: dict, env: VecEnv) -> PPO:
    device = 'cpu'
    # Map validated cfg fields to PPO constructor parameters
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



def train_model(model: PPO, cfg: dict, best_path: Path) -> int:
    total_timesteps = int(cfg["total_timesteps"])
    # Learn in rollout-sized chunks: n_envs * n_steps per iteration
    chunk = max(int(cfg["n_envs"]) * int(cfg["n_steps"]), 1)

    trained = 0
    iterations = 0

    # Configure online evaluation cadence (disabled if 0)
    eval_every_ts = int(cfg.get("eval_every_timesteps", 0) or 0)
    next_eval_ts = eval_every_ts if eval_every_ts > 0 else None

    stopper = None
    if eval_every_ts > 0:
        stopper = EarlyStopping(
            patience=5,
            min_delta=0.02,
            verbose=True,
            save_best=True,
            best_path=best_path,
        )
    else:
        stopper = None
        print("[early-stop] Disabled (eval_every_timesteps=0)")


    # Training and evaluation loop 
    while trained < total_timesteps:
        remain = total_timesteps - trained
        # Train on the next chunk (cap to remaining timesteps)
        this_chunk = min(chunk, remain)
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=(trained == 0))

        trained += this_chunk
        iterations += 1

        # Print training progress 
        print(f"[ppo] iter={iterations} ts={trained}/{total_timesteps}")

        # Trigger periodic eval once the next threshold is reached
        if next_eval_ts is not None and trained >= next_eval_ts:
            success_rate = online_eval_and_log(model, cfg, trained_ts=trained, iter_idx=iterations)
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
    parser = argparse.ArgumentParser(description="Train PPO on GridEnv")

    parser.add_argument(
        "--reward",
        choices=("base", "combined"),
        default="base",
        help="Reward type: base or combined (with potential shaping)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "training_configurations" / "ppo_config.yaml"

    cfg = load_ppo_config(config_path)
    cfg["reward_mode"] = args.reward
    print_summary(SimpleNamespace(**cfg))

    use_combined = args.reward == "combined"
    save_path = REPO_ROOT / "saved_models" / f"ppo_model_{args.reward}.zip"
    cfg["save_path"] = str(save_path)

    train_env = build_env(SimpleNamespace(**cfg), vec=True)
    train_env.env_method("set_use_combined_rewards", use_combined)  # if combined is used 
    try:
        model = build_model(cfg, train_env)
        iterations = train_model(model, cfg, save_path)
    finally:
        train_env.close()

    save_model(model, save_path)



if __name__ == "__main__":
    main()
