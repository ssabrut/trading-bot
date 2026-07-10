"""
data/pipeline.py
─────────────────────────────────────────────────────
Fetch GBPUSD M15 OHLCV bars from Dukascopy (via dukascopy-python), save to
parquet, then resample locally into H1 / H4 / D1 (no extra network fetch).

Free, cross-platform (no MT5 terminal needed), history back to 2003.
Use this for backfilling history on any OS. Live/incremental updates
still go through pipeline.py + MT5 (needs a running broker terminal).

Usage:
    python data/pipeline.py --start 2003-05-04 --end 2026-07-08
    python data/pipeline.py --start 2020-01-01 --end 2020-12-31 --instrument EUR/USD
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import dukascopy_python
import pandas as pd
from dukascopy_python import fetch as duka_fetch

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_RAW, INSTRUMENTS, MT5_SYMBOL

RESAMPLE_RULE = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def fetch_bars(instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
    raw = duka_fetch(
        instrument=instrument,
        interval=dukascopy_python.INTERVAL_MIN_15,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start,
        end=end,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {instrument} M15: {start} → {end}")

    df = raw.reset_index().rename(columns={"timestamp": "datetime"})
    df["spread"] = 0.0
    df = df[["datetime", "open", "high", "low", "close", "volume", "spread"]]
    df = df.sort_values("datetime").reset_index(drop=True)
    print(
        f"[OK] {instrument} M15: {len(df)} bars  "
        f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})"
    )
    return df


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = (
        df.set_index("datetime")
        .resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "spread": "mean",
            }
        )
    )
    return agg.dropna(subset=["open"]).reset_index()


def save_bars(df: pd.DataFrame, symbol: str, label: str):
    path = DATA_RAW / f"{symbol}_{label}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df]).drop_duplicates("datetime")
        combined = combined.sort_values("datetime").reset_index(drop=True)
        combined.to_parquet(path, index=False)
        new_rows = len(combined) - len(existing)
        print(
            f"[SAVE] Updated {path.name}: +{new_rows} new bars ({len(combined)} total)"
        )
    else:
        df.to_parquet(path, index=False)
        print(f"[SAVE] Created {path.name}: {len(df)} bars")


def run(symbol: str, start: datetime, end: datetime):
    instrument = INSTRUMENTS.get(symbol, symbol)
    m15 = fetch_bars(instrument, start, end)
    save_bars(m15, symbol, "M15")

    for label, rule in RESAMPLE_RULE.items():
        resampled = resample_bars(m15, rule)
        save_bars(resampled, symbol, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=MT5_SYMBOL)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    run(args.symbol, start_dt, end_dt)
