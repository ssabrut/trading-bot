"""
env/multi_seed.py
─────────────────────────────────────────────────────
Train the same config across multiple seeds and aggregate results — a single
seed's Sharpe/return is not reliable evidence a feature/reward change actually
helped, since PPO training variance alone can swing test-split Sharpe from
-2.5 to +0.35 on identical hyperparameters (observed empirically in this repo).

Each seed gets its own full training run + train/val/test evaluation, tagged
so it's easy to compare in MLflow. Aggregates mean±std per split/metric.

Usage:
    poetry run python env/multi_seed.py --seeds 42,123,7 --timesteps 500000 --ent-coef 0.05 --learning-rate 0.0001
    poetry run python env/multi_seed.py --seeds 42,123,7 --summary-only  # just re-aggregate existing runs
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from config.settings import ROOT
from env.evaluate import run_episode, save_run
from env.train import train as train_ppo
from env.trading_env import MultiTimeframeTradingEnv

MULTI_SEED_DIR = ROOT / "data" / "multi_seed"
SPLITS = ["train", "val", "test"]


def run_seed(seed: int, timesteps: int, ent_coef: float, learning_rate: float, batch_tag: str):
    run_tag = f"ms_{batch_tag}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model, mlflow_run_id, save_path = train_ppo(
        timesteps=timesteps,
        n_envs=1,
        seed=seed,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        run_tag=run_tag,
        experiment="trading-bot-multi-seed",
        extra_params={"batch_tag": batch_tag, "multi_seed": True},
    )

    seed_result = {"seed": seed, "mlflow_run_id": mlflow_run_id, "run_tag": run_tag, "splits": {}}
    for split in SPLITS:
        env = MultiTimeframeTradingEnv(split=split, seed=seed)
        equity_curve, trade_log, gated_steps, total_steps = run_episode(env, model=model, seed=seed)
        eval_run_id = f"{run_tag}_{split}"
        out_dir = save_run(
            eval_run_id, "GBPUSD", split, equity_curve, trade_log,
            gated_steps, total_steps, env.initial_balance,
        )
        metrics = json.loads((out_dir / "metrics.json").read_text())
        seed_result["splits"][split] = metrics
        print(f"[SEED {seed}] {split}: sharpe={metrics['return_risk']['sharpe_ratio']} return={metrics['return_risk']['total_return_pct']}%")

    return seed_result


def summarize(results: list[dict]):
    print()
    print(f"{'split':<8} {'sharpe (mean±std)':<22} {'return% (mean±std)':<24} {'max_dd% (mean±std)':<22} {'win_rate% (mean±std)'}")
    summary = {}
    for split in SPLITS:
        sharpes = [r["splits"][split]["return_risk"]["sharpe_ratio"] for r in results if r["splits"][split]["return_risk"]["sharpe_ratio"] is not None]
        returns = [r["splits"][split]["return_risk"]["total_return_pct"] for r in results]
        dds = [r["splits"][split]["return_risk"]["max_drawdown_pct"] for r in results]
        wrs = [r["splits"][split]["trade_stats"]["win_rate_pct"] for r in results if r["splits"][split]["trade_stats"]["win_rate_pct"] is not None]

        def fmt(vals):
            arr = np.array(vals)
            return f"{arr.mean():+.3f} ± {arr.std():.3f}"

        print(f"{split:<8} {fmt(sharpes):<22} {fmt(returns):<24} {fmt(dds):<22} {fmt(wrs)}")
        summary[split] = {
            "sharpe_mean": float(np.mean(sharpes)), "sharpe_std": float(np.std(sharpes)),
            "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
            "max_dd_mean": float(np.mean(dds)), "max_dd_std": float(np.std(dds)),
            "win_rate_mean": float(np.mean(wrs)), "win_rate_std": float(np.std(wrs)),
            "n_seeds": len(results),
        }

    MULTI_SEED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MULTI_SEED_DIR / "summary.json"
    out_path.write_text(json.dumps({"seeds": [r["seed"] for r in results], "summary": summary, "results": results}, indent=2))
    print(f"\n[SAVE] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated seed list")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-tag", type=str, default=None, help="Label for this batch of seeds (default: timestamp)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    batch_tag = args.batch_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for seed in seeds:
        print(f"=== training seed {seed} ({len(results)+1}/{len(seeds)}) ===")
        results.append(run_seed(seed, args.timesteps, args.ent_coef, args.learning_rate, batch_tag))

    summarize(results)
