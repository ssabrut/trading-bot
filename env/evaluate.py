"""
env/evaluate.py
─────────────────────────────────────────────────────
Runs one episode with a trained policy (or random, for smoke-testing),
dumps equity curve + trade log to data/runs/<run_id>/ for the chart UI
to render as a trade-replay overlay.

Usage:
    poetry run python env/evaluate.py --model models/ppo_trading_v1.zip --run-id ppo_v1
    poetry run python env/evaluate.py --random --run-id random_smoke   # no model needed
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
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
            "pnl": t["pnl"],
            "reason": t["reason"],
        }
        for t in trade_log
    ]
    (out_dir / "trades.json").write_text(json.dumps(trades_serializable))
    (out_dir / "meta.json").write_text(json.dumps({"symbol": symbol, "split": split, "n_trades": len(trades_serializable)}))

    print(f"[SAVE] {out_dir}: {len(equity_curve)} equity points, {len(trades_serializable)} trades")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to trained SB3 model .zip")
    parser.add_argument("--random", action="store_true", help="Use random policy (no model needed)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--symbol", type=str, default="GBPUSD")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.random and not args.model:
        parser.error("pass --model <path> or --random")

    env = MultiTimeframeTradingEnv(split=args.split, symbol=args.symbol, seed=args.seed)

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    equity_curve, trade_log = run_episode(env, model=model, seed=args.seed)
    save_run(args.run_id, args.symbol, args.split, equity_curve, trade_log)
