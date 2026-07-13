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
import numpy as np
import pandas as pd

from config.settings import ROOT
from env.trading_env import MultiTimeframeTradingEnv

RUNS_DIR = ROOT / "data" / "runs"

M15_BARS_PER_YEAR = 96 * 365  # 24h FX market, 96 M15 bars/day


def run_episode(env: MultiTimeframeTradingEnv, model=None, seed: int = 0):
    obs, info = env.reset(seed=seed)
    equity_curve = [{"time": int(pd.Timestamp(env._bars_dt[env.cursor]).timestamp()), "equity": env.balance}]
    gated_steps = 0
    total_steps = 0

    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        if env.position is None and env._regime_gate_active(env._bars_dt[env.cursor]):
            gated_steps += 1
        total_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        equity_curve.append(
            {"time": int(pd.Timestamp(env._bars_dt[min(env.cursor, len(env._bars_dt) - 1)]).timestamp()), "equity": info["equity"]}
        )
        done = terminated or truncated

    return equity_curve, info.get("trade_log", []), gated_steps, total_steps


def compute_metrics(
    equity_curve: list[dict],
    trades_serializable: list[dict],
    gated_steps: int,
    total_steps: int,
    initial_balance: float,
) -> dict:
    equity = np.array([p["equity"] for p in equity_curve], dtype=np.float64)
    step_returns = np.diff(equity) / equity[:-1]

    # return / risk
    total_return_pct = (equity[-1] - equity[0]) / equity[0] * 100
    n_periods = len(step_returns)
    years = n_periods / M15_BARS_PER_YEAR if n_periods > 0 else 0
    annualized_return_pct = (
        ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 and equity[-1] > 0 else None
    )

    ret_std = step_returns.std(ddof=1) if n_periods > 1 else 0
    sharpe_ratio = (
        (step_returns.mean() / ret_std) * np.sqrt(M15_BARS_PER_YEAR) if ret_std > 0 else None
    )
    downside = step_returns[step_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0
    sortino_ratio = (
        (step_returns.mean() / downside_std) * np.sqrt(M15_BARS_PER_YEAR) if downside_std > 0 else None
    )

    peak = np.maximum.accumulate(equity)
    drawdown_pct = (equity - peak) / peak * 100
    max_drawdown_pct = float(drawdown_pct.min()) if len(drawdown_pct) > 0 else 0.0

    # trade stats
    n_trades = len(trades_serializable)
    pnls = [t["pnl"] for t in trades_serializable]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate_pct = (len(wins) / n_trades * 100) if n_trades > 0 else None
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    avg_rr = (avg_win / abs(avg_loss)) if avg_win is not None and avg_loss not in (None, 0) else None

    max_win_streak = 0
    max_lose_streak = 0
    cur_win_streak = 0
    cur_lose_streak = 0
    for p in pnls:
        if p > 0:
            cur_win_streak += 1
            cur_lose_streak = 0
        elif p < 0:
            cur_lose_streak += 1
            cur_win_streak = 0
        else:
            cur_win_streak = 0
            cur_lose_streak = 0
        max_win_streak = max(max_win_streak, cur_win_streak)
        max_lose_streak = max(max_lose_streak, cur_lose_streak)

    longs = [t for t in trades_serializable if t["side"] == 1]
    shorts = [t for t in trades_serializable if t["side"] == -1]
    long_wins = [t for t in longs if t["pnl"] > 0]
    short_wins = [t for t in shorts if t["pnl"] > 0]
    long_short_ratio = (len(longs) / len(shorts)) if len(shorts) > 0 else None

    exit_reason_counts: dict[str, int] = {}
    for t in trades_serializable:
        exit_reason_counts[t["reason"]] = exit_reason_counts.get(t["reason"], 0) + 1

    # exposure / frequency
    span_seconds = equity_curve[-1]["time"] - equity_curve[0]["time"] if len(equity_curve) > 1 else 0
    span_days = span_seconds / 86400
    trades_per_day = (n_trades / span_days) if span_days > 0 else None
    hold_minutes = [(t["exit_time"] - t["entry_time"]) / 60 for t in trades_serializable]
    avg_hold_minutes = float(np.mean(hold_minutes)) if hold_minutes else None
    time_in_market_minutes = sum(hold_minutes)
    pct_time_in_market = (time_in_market_minutes / (span_days * 1440) * 100) if span_days > 0 else None

    return {
        "return_risk": {
            "total_return_pct": round(total_return_pct, 4),
            "annualized_return_pct": round(annualized_return_pct, 4) if annualized_return_pct is not None else None,
            "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            "sortino_ratio": round(sortino_ratio, 4) if sortino_ratio is not None else None,
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "final_equity": round(float(equity[-1]), 2),
            "initial_balance": initial_balance,
        },
        "trade_stats": {
            "n_trades": n_trades,
            "win_rate_pct": round(win_rate_pct, 2) if win_rate_pct is not None else None,
            "avg_win": round(avg_win, 2) if avg_win is not None else None,
            "avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
            "avg_rr": round(avg_rr, 4) if avg_rr is not None else None,
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "max_win_streak": max_win_streak,
            "max_lose_streak": max_lose_streak,
            "n_long": len(longs),
            "n_short": len(shorts),
            "long_short_ratio": round(long_short_ratio, 4) if long_short_ratio is not None else None,
            "long_win_rate_pct": round(len(long_wins) / len(longs) * 100, 2) if longs else None,
            "short_win_rate_pct": round(len(short_wins) / len(shorts) * 100, 2) if shorts else None,
            "exit_reason_counts": exit_reason_counts,
        },
        "exposure": {
            "span_days": round(span_days, 2),
            "trades_per_day": round(trades_per_day, 4) if trades_per_day is not None else None,
            "avg_hold_minutes": round(avg_hold_minutes, 2) if avg_hold_minutes is not None else None,
            "pct_time_in_market": round(pct_time_in_market, 2) if pct_time_in_market is not None else None,
        },
        "regime": {
            "total_steps": total_steps,
            "gated_steps": gated_steps,
            "pct_steps_gated": round(gated_steps / total_steps * 100, 2) if total_steps > 0 else None,
        },
    }


def save_run(
    run_id: str,
    symbol: str,
    split: str,
    equity_curve: list[dict],
    trade_log: list[dict],
    gated_steps: int,
    total_steps: int,
    initial_balance: float,
):
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

    metrics = compute_metrics(equity_curve, trades_serializable, gated_steps, total_steps, initial_balance)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

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
    print(
        f"[METRICS] sharpe={metrics['return_risk']['sharpe_ratio']} "
        f"return={metrics['return_risk']['total_return_pct']}% "
        f"max_dd={metrics['return_risk']['max_drawdown_pct']}% "
        f"win_rate={metrics['trade_stats']['win_rate_pct']}% "
        f"avg_rr={metrics['trade_stats']['avg_rr']} "
        f"streaks(w/l)={metrics['trade_stats']['max_win_streak']}/{metrics['trade_stats']['max_lose_streak']} "
        f"long/short={metrics['trade_stats']['n_long']}/{metrics['trade_stats']['n_short']}"
    )
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

    equity_curve, trade_log, gated_steps, total_steps = run_episode(env, model=model, seed=args.seed)
    out_dir = save_run(
        args.run_id, args.symbol, args.split, equity_curve, trade_log,
        gated_steps, total_steps, env.initial_balance,
    )

    if args.mlflow_run_id:
        metrics = json.loads((out_dir / "metrics.json").read_text())
        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_artifacts(str(out_dir), artifact_path=f"eval/{args.split}")
            for group in metrics.values():
                for key, value in group.items():
                    if isinstance(value, (int, float)) and value is not None:
                        mlflow.log_metric(f"eval_{args.split}_{key}", value)
        print(f"[MLFLOW] eval artifacts + metrics logged to run {args.mlflow_run_id} (eval/{args.split})")
