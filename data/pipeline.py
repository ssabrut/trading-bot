"""
data/pipeline.py
─────────────────────────────────────────────────────
Fetch OHLCV bars from Dukascopy (via dukascopy-python) at a configurable base
interval (default M15), save to parquet, then resample locally into higher
timeframes (no extra network fetch).

Free, cross-platform (no MT5 terminal needed), history back to 2003.
Use this for backfilling history on any OS. Live/incremental updates
still go through pipeline.py + MT5 (needs a running broker terminal).

Usage:
    python data/pipeline.py --start 2003-05-04 --end 2026-07-08
    python data/pipeline.py --start 2020-01-01 --end 2020-12-31 --instrument EUR/USD
    python data/pipeline.py --symbol NAS100 --start 2012-01-01 --end 2026-07-14 --base-interval M1
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

BASE_INTERVALS = {
    "M1": dukascopy_python.INTERVAL_MIN_1,
    "M15": dukascopy_python.INTERVAL_MIN_15,
}

# resample targets, keyed by base interval — only include TFs coarser than the base
RESAMPLE_RULE = {
    "M1": {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"},
    "M15": {"H1": "1h", "H4": "4h", "D1": "1D"},
}


def fetch_bars(instrument: str, start: datetime, end: datetime, base_interval: str) -> pd.DataFrame:
    raw = duka_fetch(
        instrument=instrument,
        interval=BASE_INTERVALS[base_interval],
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start,
        end=end,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {instrument} {base_interval}: {start} → {end}")

    df = raw.reset_index().rename(columns={"timestamp": "datetime"})
    df["spread"] = 0.0
    df = df[["datetime", "open", "high", "low", "close", "volume", "spread"]]
    df = df.sort_values("datetime").reset_index(drop=True)
    print(
        f"[OK] {instrument} {base_interval}: {len(df)} bars  "
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


def run(symbol: str, start: datetime, end: datetime, base_interval: str):
    instrument = INSTRUMENTS.get(symbol, symbol)
    base = fetch_bars(instrument, start, end, base_interval)
    save_bars(base, symbol, base_interval)

    for label, rule in RESAMPLE_RULE[base_interval].items():
        resampled = resample_bars(base, rule)
        save_bars(resampled, symbol, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=MT5_SYMBOL)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--base-interval", default="M15", choices=list(BASE_INTERVALS), help="Base fetch granularity; higher TFs are resampled from this locally")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    run(args.symbol, start_dt, end_dt, args.base_interval)
