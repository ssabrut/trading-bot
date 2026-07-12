"""
env/train.py
─────────────────────────────────────────────────────
PPO baseline (MultiInputPolicy = MLP per obs key, concat) against
MultiTimeframeTradingEnv. v1: verify env/reward correctness before
swapping in a custom Transformer feature extractor.

Usage:
    poetry run python env/train.py --timesteps 50000
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config.settings import ROOT
from env.trading_env import MultiTimeframeTradingEnv

MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def make_env(split: str, seed: int):
    def _init():
        env = MultiTimeframeTradingEnv(split=split, seed=seed)
        return Monitor(env)
    return _init


def train(timesteps: int, n_envs: int, seed: int):
    vec_env = make_vec_env(
        make_env("train", seed),
        n_envs=n_envs,
        seed=seed,
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(LOGS_DIR / "ppo_tensorboard"),
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)

    save_path = MODELS_DIR / "ppo_trading_v1.zip"
    model.save(save_path)
    print(f"[SAVE] {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.seed)
