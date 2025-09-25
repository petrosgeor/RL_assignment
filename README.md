# RL Assignment — Grid Environments, Training, and Evaluation

Train PPO (on‑policy) and DQN (off‑policy) agents on a custom grid‑world Gymnasium environment, then evaluate over 100 episodes reporting success rate, average return, and average steps. The repo includes a base environment and an expanded variant with first‑visit bonuses.

## Table of Contents
- Getting Started
- Project Structure
- Environments
- Training
- Evaluation
- Observe a Single Episode
- Configuration Reference


## Getting Started
```bash
conda create -n rl_assignment python=3.10
conda activate rl_assignment
pip install -r requirements.txt
```

## Project Structure
- environment.py (base GridEnv)
- expanded_environment.py (ExpandedGridEnv with first‑visit bonuses)
- training/
  - trainer_ppo.py (PPO on base env)
  - trainer_dqn.py (DQN on base env)
  - expanded_training.py (PPO on expanded env)
  - training_evaluation.py (shared evaluation helpers)
  - utils.py (env builders and utilities)
- evaluation/evaluation.py (evaluation CLI)
- training_configurations/
  - ppo_config.yaml
  - dqn_config.yaml
  - env_config.yaml (default 6×6 grid, 5 obstacles)

## Environments
- Base environment (environment.py): 3‑layer observation (agent, goal, obstacles) with 4 actions. Episodes end on goal, collision, or out‑of‑bounds. Potential‑based shaping can be enabled via reward combined in trainers/evaluator.
- Expanded environment (expanded_environment.py): adds a bonuses layer (4‑layer observation). Goal only pays when all first‑visit bonuses are collected.
- Defaults come from training_configurations/env_config.yaml: grid_rows 6, grid_cols 6, num_obstacles 5.

## Training
- PPO on base env
  ```bash
  # reward: base | combined
  python training/trainer_ppo.py --reward base
  ```
  Loads hyperparameters from training_configurations/ppo_config.yaml and saves to saved_models/ppo_model_<reward>.zip.

- DQN on base env
  ```bash
  # reward: base | combined
  python training/trainer_dqn.py --reward base
  ```
  Loads hyperparameters from training_configurations/dqn_config.yaml and saves to saved_models/dqn_model_<reward>.zip.

- PPO on expanded env
  ```bash
  python training/expanded_training.py
  ```
  Uses training_configurations/ppo_config.yaml and saves best to saved_models/ppo_model_expanded.zip (via early stopping).

## Evaluation
Use a single CLI for both algorithms and environments.
```bash
# Base environment (PPO or DQN). Choose the reward mode used in training.
python evaluation/evaluation.py --model ppo --env base --reward base
python evaluation/evaluation.py --model dqn --env base --reward combined

# Expanded environment (PPO only)
python evaluation/evaluation.py --model ppo --env expanded
```
Outputs include average_return, average_length, and success_rate (expanded also reports average_bonuses_visited).

Expected checkpoints:
- Base PPO: saved_models/ppo_model_<base|combined>.zip
- Base DQN: saved_models/dqn_model_<base|combined>.zip
- Expanded PPO: saved_models/ppo_model_expanded.zip

## Observe a Single Episode
No extra script required. Example to render one PPO episode on the base env:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from environment import load_grid_config, create_env_from_config

cfg = load_grid_config("training_configurations/env_config.yaml")
env = create_env_from_config(cfg)
model = PPO.load("saved_models/ppo_model_base.zip", device="cpu")

obs, _ = env.reset(seed=0)
done = False
while not done:
  env.render(mode="human")
  action, _ = model.predict(obs, deterministic=True)
  obs, reward, terminated, truncated, info = env.step(int(action))
  done = terminated or truncated
```
The same approach works for DQN (import DQN and load the matching checkpoint). To use the expanded environment, import from expanded_environment and load the expanded PPO model.

## Configuration Reference
- training_configurations/ppo_config.yaml: PPO hyperparameters (policy, n_steps, batch_size, learning_rate, gamma, etc.).
- training_configurations/dqn_config.yaml: DQN hyperparameters (buffer_size, train_freq, gradient_steps, target_update_interval, etc.).
- training_configurations/env_config.yaml: grid_rows, grid_cols, num_obstacles; also used to derive max_episode_steps and flatten settings via training YAMLs.

