"""
env/evaluate.py
─────────────────────────────────────────────────────
Runs one episode with a trained policy (or random, for smoke-testing),
dumps equity curve + trade log to data/runs/<run_id>/ for the chart UI
to render as a trade-replay overlay.

If --mlflow-run-id is given, the model artifact is pulled from that MLflow
run (logged by train.py) instead of a local path, and the equity/trade JSON
is logged back onto the same run as eval artifacts.

Usage:
    poetry run python env/evaluate.py --model models/ppo_<run_id>.zip --run-id ppo_v1
    poetry run python env/evaluate.py --mlflow-run-id <run_id> --run-id ppo_v1
    poetry run python env/evaluate.py --random --run-id random_smoke   # no model needed
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import pandas as pd

from config.settings import ROOT
from env.trading_env import MultiTimeframeTradingEnv

RUNS_DIR = ROOT / "data" / "runs"


def run_episode(env: MultiTimeframeTradingEnv, model=None, seed: int = 0):
    obs, info = env.reset(seed=seed)
    equity_curve = [{"time": int(pd.Timestamp(env._bars_dt[env.cursor]).timestamp()), "equity": env.balance}]

    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        equity_curve.append(
            {"time": int(pd.Timestamp(env._bars_dt[min(env.cursor, len(env._bars_dt) - 1)]).timestamp()), "equity": info["equity"]}
        )
        done = terminated or truncated

    return equity_curve, info.get("trade_log", [])


def save_run(run_id: str, symbol: str, split: str, equity_curve: list[dict], trade_log: list[dict]):
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "equity.json").write_text(json.dumps(equity_curve))

    trades_serializable = [
        {
            "entry_time": int(pd.Timestamp(t["entry_time"]).timestamp()),
            "entry_price": t["entry_price"],
            "exit_time": int(pd.Timestamp(t["exit_time"]).timestamp()),
            "exit_price": t["exit_price"],
            "side": t["side"],
            "size": t["size"],
            "sl": t["sl"],
            "tp": t["tp"],
            "pnl": t["pnl"],
            "reason": t["reason"],
        }
        for t in trade_log
    ]
    (out_dir / "trades.json").write_text(json.dumps(trades_serializable))
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "split": split,
                "n_trades": len(trades_serializable),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    )

    print(f"[SAVE] {out_dir}: {len(equity_curve)} equity points, {len(trades_serializable)} trades")
    return out_dir


def load_model_from_mlflow(mlflow_run_id: str, best: bool = False):
    from stable_baselines3 import PPO

    artifact_path = "model_best_val" if best else "model"
    local_dir = mlflow.artifacts.download_artifacts(run_id=mlflow_run_id, artifact_path=artifact_path)
    zips = list(Path(local_dir).glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No model .zip found in MLflow run {mlflow_run_id} artifact_path='{artifact_path}'")
    return PPO.load(zips[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to trained SB3 model .zip")
    parser.add_argument("--mlflow-run-id", type=str, default=None, help="Pull model artifact from this MLflow run instead")
    parser.add_argument("--best", action="store_true", help="With --mlflow-run-id: pull the best-on-val checkpoint (model_best_val) instead of the final model")
    parser.add_argument("--random", action="store_true", help="Use random policy (no model needed)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--symbol", type=str, default="GBPUSD")
    parser.add_argument("--run-id", type=str, default=None, help="Local run-id — names the data/runs/<run-id>/ output folder (default: current timestamp)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.random and not args.model and not args.mlflow_run_id:
        parser.error("pass --model <path>, --mlflow-run-id <id>, or --random")

    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = MultiTimeframeTradingEnv(split=args.split, symbol=args.symbol, seed=args.seed)

    model = None
    if args.mlflow_run_id:
        model = load_model_from_mlflow(args.mlflow_run_id, best=args.best)
    elif args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    equity_curve, trade_log = run_episode(env, model=model, seed=args.seed)
    out_dir = save_run(args.run_id, args.symbol, args.split, equity_curve, trade_log)

    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_artifacts(str(out_dir), artifact_path=f"eval/{args.split}")
            mlflow.log_metric(f"eval_{args.split}_n_trades", len(trade_log))
            mlflow.log_metric(f"eval_{args.split}_final_equity", equity_curve[-1]["equity"])
        print(f"[MLFLOW] eval artifacts logged to run {args.mlflow_run_id} (eval/{args.split})")
