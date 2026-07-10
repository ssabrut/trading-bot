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

from config.settings import DATA_RAW, MT5_SYMBOL

app = FastAPI(title="trading-bot chart viewer")

STATIC_DIR = Path(__file__).parent / "static"
TIMEFRAMES = ["M15", "H1", "H4", "D1"]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/timeframes")
def list_timeframes():
    available = [
        tf for tf in TIMEFRAMES if (DATA_RAW / f"{MT5_SYMBOL}_{tf}.parquet").exists()
    ]
    return {"symbol": MT5_SYMBOL, "timeframes": available}


@app.get("/api/bars")
def get_bars(tf: str = "M15"):
    if tf not in TIMEFRAMES:
        raise HTTPException(
            400, f"Unknown timeframe '{tf}', expected one of {TIMEFRAMES}"
        )

    path = DATA_RAW / f"{MT5_SYMBOL}_{tf}.parquet"
    if not path.exists():
        raise HTTPException(
            404, f"No data for {MT5_SYMBOL} {tf}. Run data/pipeline.py first."
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
