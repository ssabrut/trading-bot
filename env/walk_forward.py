"""
env/walk_forward.py
─────────────────────────────────────────────────────
Expanding-window walk-forward validation. Each fold trains on all history up
to a cutoff year and tests on the following year — train window grows fold
to fold (matches how you'd actually retrain in production), test window is
a held-out 1-year slice never seen during that fold's training.

Fold N: train [2012-01-11, {FOLD_YEARS[N]}-01-01)  ->  test [{FOLD_YEARS[N]}-01-01, {FOLD_YEARS[N]+1}-01-01)

Each fold gets its own scaler (fit on that fold's train window only — no
leakage from future folds) and its own normalized data, written under
data/walk_forward/fold_<N>/ so folds never clobber each other or the main
pipeline's data/splits, data/normalized, config/scalers.

Usage:
    poetry run python env/walk_forward.py --fold 1 --timesteps 500000
    poetry run python env/walk_forward.py --summary   # aggregate all completed folds
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_FEATURES, ROOT
from env.evaluate import run_episode, save_run
from env.train import train as train_ppo
from env.trading_env import FEATURE_COLS, MultiTimeframeTradingEnv, TIMEFRAMES

SYMBOL = "GBPUSD"
DATA_START = pd.Timestamp("2012-01-11", tz="UTC")

# fold N: train < FOLD_YEARS[N-1], test [FOLD_YEARS[N-1], FOLD_YEARS[N-1]+1)
FOLD_TEST_START_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

WF_DIR = ROOT / "data" / "walk_forward"

SKIP_COLS = {
    "datetime",
    "ema_cross_up", "ema_cross_down",
    "ema_regime", "ema_trend",
    "session_asian", "session_london", "session_ny", "session_overlap",
    "hour_sin", "hour_cos",
}


VAL_MONTHS = 6  # last N months of each fold's train window carved out as val (checkpoint selection)


def fold_dates(fold: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """1-indexed fold -> (train_start, val_start, test_start, test_end).
    train: [train_start, val_start) | val: [val_start, test_start) | test: [test_start, test_end)"""
    if not (1 <= fold <= len(FOLD_TEST_START_YEARS)):
        raise ValueError(f"fold must be in [1, {len(FOLD_TEST_START_YEARS)}]")
    test_start = pd.Timestamp(f"{FOLD_TEST_START_YEARS[fold - 1]}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{FOLD_TEST_START_YEARS[fold - 1] + 1}-01-01", tz="UTC")
    val_start = test_start - pd.DateOffset(months=VAL_MONTHS)
    return DATA_START, val_start, test_start, test_end


def fold_dir(fold: int) -> Path:
    return WF_DIR / f"fold_{fold}"


def split_and_normalize_fold(fold: int):
    """Re-split + re-normalize per-TF features for this fold's train/val/test windows.
    train: [train_start, val_start) | val: [val_start, test_start) | test: [test_start, test_end)
    Scaler fit on this fold's train split only (val/test never seen by the scaler either)."""
    train_start, val_start, test_start, test_end = fold_dates(fold)
    out_dir = fold_dir(fold)
    splits_dir = out_dir / "splits"
    norm_dir = out_dir / "normalized"
    splits_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    scalers = {}
    for tf in TIMEFRAMES:
        df = pd.read_parquet(DATA_FEATURES / f"{SYMBOL}_{tf}_features.parquet")
        dt = df["datetime"]

        train_df = df[(dt >= train_start) & (dt < val_start)].reset_index(drop=True)
        val_df = df[(dt >= val_start) & (dt < test_start)].reset_index(drop=True)
        test_df = df[(dt >= test_start) & (dt < test_end)].reset_index(drop=True)

        scale_cols = [c for c in train_df.columns if c not in SKIP_COLS]
        scaler = {}
        for col in scale_cols:
            median = train_df[col].median()
            q75, q25 = train_df[col].quantile([0.75, 0.25])
            iqr = q75 - q25 if q75 - q25 != 0 else 1.0
            scaler[col] = {"median": float(median), "iqr": float(iqr)}
        scalers[tf] = {"cols": scale_cols, "params": scaler}

        for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
            part.to_parquet(splits_dir / f"{SYMBOL}_{tf}_{name}.parquet", index=False)
            normed = part.copy()
            for col in scale_cols:
                normed[col] = (part[col] - scaler[col]["median"]) / scaler[col]["iqr"]
            normed.to_parquet(norm_dir / f"{SYMBOL}_{tf}_{name}.parquet", index=False)

        print(
            f"[FOLD {fold}] {tf}: train {len(train_df)} rows ({train_start.date()}..{val_start.date()}), "
            f"val {len(val_df)} rows ({val_start.date()}..{test_start.date()}), "
            f"test {len(test_df)} rows ({test_start.date()}..{test_end.date()})"
        )

    (out_dir / "scalers.json").write_text(json.dumps(scalers, default=str))
    return norm_dir, splits_dir


def run_fold(fold: int, timesteps: int, ent_coef: float, learning_rate: float, seeds: list[int]):
    train_start, val_start, test_start, test_end = fold_dates(fold)
    norm_dir, splits_dir = split_and_normalize_fold(fold)  # scaler is seed-independent, computed once

    seed_results = []
    for seed in seeds:
        run_tag = f"wf_fold{fold}_seed{seed}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        model, run_id, save_path = train_ppo(
            timesteps=timesteps,
            n_envs=1,
            seed=seed,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            run_tag=run_tag,
            normalized_dir=norm_dir,
            train_date_range=(train_start, val_start),
            eval_date_range=(val_start, test_start),
            experiment="trading-bot-walk-forward",
            extra_params={
                "fold": fold,
                "seed": seed,
                "train_start": str(train_start.date()),
                "val_start": str(val_start.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
            },
        )

        # final held-out evaluation on this fold's test window — never seen during
        # training or checkpoint selection (train/val/test are properly disjoint, see fold_dates)
        env = MultiTimeframeTradingEnv(
            split="test",
            symbol=SYMBOL,
            seed=seed,
            normalized_dir=norm_dir,
            date_range=(test_start, test_end),
        )
        equity_curve, trade_log, gated_steps, total_steps = run_episode(env, model=model, seed=seed)

        fold_run_id = f"wf_fold{fold}_{run_tag}"
        out_dir = save_run(
            fold_run_id, SYMBOL, "test", equity_curve, trade_log,
            gated_steps, total_steps, env.initial_balance,
        )

        print(f"[FOLD {fold} SEED {seed}] done. mlflow_run_id={run_id}  eval_run_id={fold_run_id}")
        seed_results.append({"seed": seed, "mlflow_run_id": run_id, "run_id": fold_run_id})

    fold_summary = {
        "fold": fold,
        "train_start": str(train_start.date()),
        "val_start": str(val_start.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "seed_results": seed_results,
    }
    (fold_dir(fold) / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2))

    print(f"[FOLD {fold}] done, {len(seeds)} seed(s).")
    return fold_summary


def summarize():
    rows = []
    running_equity = None  # compounds fold-to-fold (using each fold's MEAN return across seeds)
    for fold in range(1, len(FOLD_TEST_START_YEARS) + 1):
        summary_path = fold_dir(fold) / "fold_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())

        seed_results = summary.get("seed_results")
        if seed_results is None:
            # legacy single-seed fold_summary.json (pre multi-seed support)
            seed_results = [{"seed": summary.get("seed"), "run_id": summary["run_id"]}]

        seed_metrics = []
        for sr in seed_results:
            metrics_path = ROOT / "data" / "runs" / sr["run_id"] / "metrics.json"
            if metrics_path.exists():
                seed_metrics.append(json.loads(metrics_path.read_text()))
        if not seed_metrics:
            continue

        sharpes = np.array([m["return_risk"]["sharpe_ratio"] for m in seed_metrics if m["return_risk"]["sharpe_ratio"] is not None])
        returns = np.array([m["return_risk"]["total_return_pct"] for m in seed_metrics])
        dds = np.array([m["return_risk"]["max_drawdown_pct"] for m in seed_metrics])
        wrs = np.array([m["trade_stats"]["win_rate_pct"] for m in seed_metrics if m["trade_stats"]["win_rate_pct"] is not None])
        start_equities = np.array([m["return_risk"]["initial_balance"] for m in seed_metrics])
        end_equities = np.array([m["return_risk"]["final_equity"] for m in seed_metrics])

        mean_start_equity = float(start_equities.mean())
        mean_end_equity = float(end_equities.mean())
        fold_return_frac = (mean_end_equity - mean_start_equity) / mean_start_equity

        if running_equity is None:
            running_equity = mean_start_equity
        running_equity_start = running_equity
        running_equity *= 1 + fold_return_frac  # apply this fold's mean % return to the running balance

        rows.append(
            {
                "fold": fold,
                "test_window": f"{summary['test_start']}..{summary['test_end']}",
                "n_seeds": len(seed_metrics),
                "sharpe_mean": round(float(sharpes.mean()), 4) if len(sharpes) else None,
                "sharpe_std": round(float(sharpes.std()), 4) if len(sharpes) else None,
                "return_pct_mean": round(float(returns.mean()), 4),
                "return_pct_std": round(float(returns.std()), 4),
                "max_dd_pct_mean": round(float(dds.mean()), 4),
                "win_rate_pct_mean": round(float(wrs.mean()), 4) if len(wrs) else None,
                "start_equity": round(mean_start_equity, 2),
                "end_equity": round(mean_end_equity, 2),
                "running_equity_start": round(running_equity_start, 2),
                "running_equity_end": round(running_equity, 2),
            }
        )

    if not rows:
        print("No completed folds found.")
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print("Aggregate (mean across folds, using each fold's mean-across-seeds):")
    print(df[["sharpe_mean", "return_pct_mean", "max_dd_pct_mean", "win_rate_pct_mean"]].mean())
    print()
    n_years = len(df)
    start_eq = df['running_equity_start'].iloc[0]
    end_eq = df['running_equity_end'].iloc[-1]
    total_compounded_return = (end_eq / start_eq - 1) * 100
    cagr = ((end_eq / start_eq) ** (1 / n_years) - 1) * 100 if start_eq > 0 else None
    print(f"Compounded equity if traded continuously across all folds' years (using mean-across-seeds per fold):")
    print(f"  start: {start_eq:.2f}  ->  end: {end_eq:.2f}")
    print(f"  total compounded return: {total_compounded_return:+.2f}%  over {n_years} years")
    print(f"  CAGR (compound annual growth rate): {cagr:+.2f}%/yr" if cagr is not None else "  CAGR: n/a")

    (WF_DIR / "summary.json").write_text(df.to_json(orient="records", indent=2))
    print(f"\n[SAVE] {WF_DIR / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help=f"Fold number, 1-{len(FOLD_TEST_START_YEARS)}")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seed list, e.g. 42,123,7 — one full train+eval per seed, results averaged")
    parser.add_argument("--summary", action="store_true", help="Aggregate all completed folds and print/save summary")
    args = parser.parse_args()

    if args.summary:
        summarize()
    elif args.fold is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
        run_fold(args.fold, args.timesteps, args.ent_coef, args.learning_rate, seeds)
    else:
        parser.error("pass --fold N or --summary")
