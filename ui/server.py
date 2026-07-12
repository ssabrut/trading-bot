"""
ui/server.py
─────────────────────────────────────────────────────
Local chart server. Serves OHLCV bars from data/raw/*.parquet as JSON
and a TradingView-style candlestick UI (lightweight-charts) to view them.

Usage:
    poetry run uvicorn ui.server:app --reload
    open http://127.0.0.1:8000
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import DATA_FEATURES, DATA_LABELS, DATA_PROCESSED, DATA_RAW, MT5_SYMBOL

app = FastAPI(title="trading-bot chart viewer")

STATIC_DIR = Path(__file__).parent / "static"
TIMEFRAMES = ["M15", "H1", "H4", "D1"]
SOURCES = {"raw": DATA_RAW, "processed": DATA_PROCESSED}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/timeframes")
def list_timeframes(source: str = "raw"):
    if source not in SOURCES:
        raise HTTPException(400, f"Unknown source '{source}', expected one of {list(SOURCES)}")
    base = SOURCES[source]
    available = [
        tf for tf in TIMEFRAMES if (base / f"{MT5_SYMBOL}_{tf}.parquet").exists()
    ]
    return {"symbol": MT5_SYMBOL, "timeframes": available}


@app.get("/api/bars")
def get_bars(tf: str = "M15", source: str = "raw"):
    if tf not in TIMEFRAMES:
        raise HTTPException(
            400, f"Unknown timeframe '{tf}', expected one of {TIMEFRAMES}"
        )
    if source not in SOURCES:
        raise HTTPException(400, f"Unknown source '{source}', expected one of {list(SOURCES)}")

    path = SOURCES[source] / f"{MT5_SYMBOL}_{tf}.parquet"
    if not path.exists():
        raise HTTPException(
            404, f"No data for {MT5_SYMBOL} {tf} in '{source}'. Run data/pipeline.py first."
        )

    df = pd.read_parquet(path)
    bars = [
        {
            "time": int(row.datetime.timestamp()),
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
        }
        for row in df.itertuples()
    ]
    return {"symbol": MT5_SYMBOL, "timeframe": tf, "bars": bars}


@app.get("/api/swings")
def get_swings(tf: str = "M15"):
    if tf not in TIMEFRAMES:
        raise HTTPException(
            400, f"Unknown timeframe '{tf}', expected one of {TIMEFRAMES}"
        )

    path = DATA_FEATURES / f"{MT5_SYMBOL}_{tf}_features.parquet"
    if not path.exists():
        return {"symbol": MT5_SYMBOL, "timeframe": tf, "highs": [], "lows": []}

    df = pd.read_parquet(path)
    if "swing_high" not in df.columns or "swing_low" not in df.columns:
        return {"symbol": MT5_SYMBOL, "timeframe": tf, "highs": [], "lows": []}

    ohlc_path = DATA_PROCESSED / f"{MT5_SYMBOL}_{tf}.parquet"
    price = pd.read_parquet(ohlc_path)[["datetime", "high", "low"]]
    merged = df[["datetime", "swing_high", "swing_low"]].merge(price, on="datetime", how="left")

    highs = [
        {"time": int(row.datetime.timestamp()), "price": row.high}
        for row in merged[merged["swing_high"] == 1].itertuples()
    ]
    lows = [
        {"time": int(row.datetime.timestamp()), "price": row.low}
        for row in merged[merged["swing_low"] == 1].itertuples()
    ]
    return {"symbol": MT5_SYMBOL, "timeframe": tf, "highs": highs, "lows": lows}


@app.get("/api/labels")
def get_labels(tf: str = "M15"):
    if tf not in TIMEFRAMES:
        raise HTTPException(
            400, f"Unknown timeframe '{tf}', expected one of {TIMEFRAMES}"
        )

    path = DATA_LABELS / f"{MT5_SYMBOL}_{tf}_labels.parquet"
    if not path.exists():
        return {"symbol": MT5_SYMBOL, "timeframe": tf, "events": []}

    df = pd.read_parquet(path)
    events = [
        {
            "entry_time": int(row.datetime.timestamp()),
            "entry_price": row.entry_price,
            "exit_time": int(row.exit_datetime.timestamp()),
            "exit_price": row.exit_price,
            "upper_barrier": row.upper_barrier,
            "lower_barrier": row.lower_barrier,
            "label": int(row.label),
        }
        for row in df.itertuples()
    ]
    return {"symbol": MT5_SYMBOL, "timeframe": tf, "events": events}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
