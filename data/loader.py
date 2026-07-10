"""
data/loader.py
─────────────────────────────────────────────────────
Load raw OHLCV parquet bars for a given symbol + timeframe(s) from
data/raw/{symbol}_{label}.parquet (produced by pipeline.py).

Usage:
    from data.loader import load_bars, load_multi_tf

    h1 = load_bars("GBPUSD", "H1")
    tfs = load_multi_tf("GBPUSD", ["D1", "H4", "H1", "M15"])
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_RAW

VALID_TIMEFRAMES = ("M15", "H1", "H4", "D1")


def load_bars(
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load one timeframe's OHLCV parquet for `symbol`, optionally sliced to [start, end]."""
    timeframe = timeframe.upper()
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(f"Unknown timeframe '{timeframe}', expected one of {VALID_TIMEFRAMES}")

    path = DATA_RAW / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No parquet found at {path}")

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    if start is not None:
        df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["datetime"] <= pd.Timestamp(end, tz="UTC")]

    return df.reset_index(drop=True)


def load_multi_tf(
    symbol: str,
    timeframes: list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load multiple timeframes for `symbol`, keyed by timeframe label."""
    return {tf.upper(): load_bars(symbol, tf, start=start, end=end) for tf in timeframes}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="GBPUSD")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(VALID_TIMEFRAMES),
        choices=VALID_TIMEFRAMES,
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    data = load_multi_tf(args.symbol, args.timeframes, start=args.start, end=args.end)
    for tf, df in data.items():
        print(f"{tf}: {len(df)} bars  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")
