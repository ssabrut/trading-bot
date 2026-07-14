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
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config.settings import ROOT
from env.trading_env import (
    ADX_MIN_THRESHOLD,
    DOWNSIDE_PENALTY_COEF,
    EPISODE_DAYS_DEFAULT,
    FEATURE_COLS,
    IDLE_PENALTY,
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
EVAL_FREQ = 10_000  # env steps between deterministic val-split rollouts
EVAL_EPISODES = 3


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


def make_env(split: str, seed: int, normalized_dir=None, date_range=None):
    def _init():
        env = MultiTimeframeTradingEnv(
            split=split, seed=seed, normalized_dir=normalized_dir, date_range=date_range,
        )
        return Monitor(env)
    return _init


def train(
    timesteps: int,
    n_envs: int,
    seed: int,
    ent_coef: float,
    learning_rate: float,
    run_tag: str | None = None,
    normalized_dir=None,
    train_date_range=None,
    eval_date_range=None,
    experiment: str = MLFLOW_EXPERIMENT,
    extra_params: dict | None = None,
):
    mlflow.set_experiment(experiment)  # tracking URI picked up from MLFLOW_TRACKING_URI env var

    run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")  # sortable local name — MLflow's own run_id stays the uuid

    vec_env = make_vec_env(
        make_env("train", seed, normalized_dir=normalized_dir, date_range=train_date_range),
        n_envs=n_envs,
        seed=seed,
    )
    eval_env = make_vec_env(
        make_env("val", seed, normalized_dir=normalized_dir, date_range=eval_date_range),
        n_envs=1,
        seed=seed,
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        tensorboard_log=str(LOGS_DIR / "ppo_tensorboard"),
    )

    with mlflow.start_run(run_name=run_tag) as run:
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "run_tag": run_tag,
                "timesteps": timesteps,
                "n_envs": n_envs,
                "seed": seed,
                "ent_coef": ent_coef,
                "learning_rate": learning_rate,
                "episode_days": EPISODE_DAYS_DEFAULT,
                "sl_atr_mult": SL_ATR_MULT,
                "tp_atr_mult": TP_ATR_MULT,
                "risk_per_trade": RISK_PER_TRADE,
                "idle_penalty": IDLE_PENALTY,
                "downside_penalty_coef": DOWNSIDE_PENALTY_COEF,
                "adx_min_threshold": ADX_MIN_THRESHOLD,
                "window_m15": WINDOW["M15"],
                "window_h1": WINDOW["H1"],
                "window_h4": WINDOW["H4"],
                "window_d1": WINDOW["D1"],
                "feature_cols_m15": ",".join(FEATURE_COLS["M15"]),
                "feature_cols_h1": ",".join(FEATURE_COLS["H1"]),
                "feature_cols_h4": ",".join(FEATURE_COLS["H4"]),
                "feature_cols_d1": ",".join(FEATURE_COLS["D1"]),
                "eval_freq": EVAL_FREQ,
                "eval_episodes": EVAL_EPISODES,
                **(extra_params or {}),
            }
        )

        best_model_dir = MODELS_DIR / f"ppo_{run_tag}_best"
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(LOGS_DIR / "eval" / run_tag),
            eval_freq=max(EVAL_FREQ // n_envs, 1),
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            verbose=1,
        )

        model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            callback=[eval_callback, MLflowCallback()],
        )

        save_path = MODELS_DIR / f"ppo_{run_tag}.zip"
        model.save(save_path)
        print(f"[SAVE] {save_path}")

        mlflow.log_artifact(str(save_path), artifact_path="model")

        best_model_path = best_model_dir / "best_model.zip"
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="model_best_val")
            print(f"[SAVE] {best_model_path} (best on val split during training)")

        scaler_dir = ROOT / "config" / "scalers"
        if scaler_dir.exists():
            mlflow.log_artifacts(str(scaler_dir), artifact_path="scalers")

        print(f"[MLFLOW] run_id={run_id}  run_tag={run_tag}  experiment={experiment}")

    return model, run_id, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="SB3 PPO default is 3e-4; lower (1e-4–2e-4) slows convergence and reduces overfitting on long runs")
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.seed, args.ent_coef, args.learning_rate)
