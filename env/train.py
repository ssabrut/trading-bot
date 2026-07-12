"""
env/train.py
─────────────────────────────────────────────────────
PPO baseline (MultiInputPolicy = MLP per obs key, concat) against
MultiTimeframeTradingEnv. v1: verify env/reward correctness before
swapping in a custom Transformer feature extractor.

Logs params/metrics/model to MLflow (backed by MinIO S3 for artifacts) —
requires MLFLOW_TRACKING_URI / AWS_* env vars set (see .env.example) and
the mlflow stack running (`make mlflow-up`).

Usage:
    poetry run python env/train.py --timesteps 50000
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config.settings import ROOT
from env.trading_env import (
    EPISODE_DAYS_DEFAULT,
    FEATURE_COLS,
    RISK_PER_TRADE,
    SL_ATR_MULT,
    TP_ATR_MULT,
    WINDOW,
    MultiTimeframeTradingEnv,
)

MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_EXPERIMENT = "trading-bot-ppo"
METRIC_LOG_FREQ = 1000  # env steps between MLflow metric flushes


class MLflowCallback(BaseCallback):
    def __init__(self, log_freq: int = METRIC_LOG_FREQ, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            for key, value in self.logger.name_to_value.items():
                try:
                    mlflow.log_metric(key.replace("/", "_"), float(value), step=self.num_timesteps)
                except (TypeError, ValueError):
                    continue
        return True


def make_env(split: str, seed: int):
    def _init():
        env = MultiTimeframeTradingEnv(split=split, seed=seed)
        return Monitor(env)
    return _init


def train(timesteps: int, n_envs: int, seed: int, ent_coef: float):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)  # tracking URI picked up from MLFLOW_TRACKING_URI env var

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
        ent_coef=ent_coef,
        tensorboard_log=str(LOGS_DIR / "ppo_tensorboard"),
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "timesteps": timesteps,
                "n_envs": n_envs,
                "seed": seed,
                "ent_coef": ent_coef,
                "episode_days": EPISODE_DAYS_DEFAULT,
                "sl_atr_mult": SL_ATR_MULT,
                "tp_atr_mult": TP_ATR_MULT,
                "risk_per_trade": RISK_PER_TRADE,
                "window_m15": WINDOW["M15"],
                "window_h1": WINDOW["H1"],
                "window_h4": WINDOW["H4"],
                "window_d1": WINDOW["D1"],
                "feature_cols_m15": ",".join(FEATURE_COLS["M15"]),
                "feature_cols_h1": ",".join(FEATURE_COLS["H1"]),
                "feature_cols_h4": ",".join(FEATURE_COLS["H4"]),
                "feature_cols_d1": ",".join(FEATURE_COLS["D1"]),
            }
        )

        model.learn(total_timesteps=timesteps, progress_bar=True, callback=MLflowCallback())

        save_path = MODELS_DIR / f"ppo_{run_id}.zip"
        model.save(save_path)
        print(f"[SAVE] {save_path}")

        mlflow.log_artifact(str(save_path), artifact_path="model")

        scaler_dir = ROOT / "config" / "scalers"
        if scaler_dir.exists():
            mlflow.log_artifacts(str(scaler_dir), artifact_path="scalers")

        print(f"[MLFLOW] run_id={run_id}  experiment={MLFLOW_EXPERIMENT}")

    return model, run_id, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.seed, args.ent_coef)
