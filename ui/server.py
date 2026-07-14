"""
ui/server.py
─────────────────────────────────────────────────────
Local chart server. Serves OHLCV bars from data/raw/*.parquet as JSON
and a TradingView-style candlestick UI (lightweight-charts) to view them.

Usage:
    poetry run uvicorn ui.server:app --reload
    open http://127.0.0.1:8000
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import DATA_FEATURES, DATA_PROCESSED, DATA_RAW, MT5_SYMBOL, ROOT

app = FastAPI(title="trading-bot chart viewer")

STATIC_DIR = Path(__file__).parent / "static"
TIMEFRAMES = ["M15", "H1", "H4", "D1"]
SOURCES = {"raw": DATA_RAW, "processed": DATA_PROCESSED}
RUNS_DIR = ROOT / "data" / "runs"


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


# indicator columns to expose per TF, grouped by chart placement.
# "overlay" columns share the price scale; "oscillator" columns get their own sub-pane.
INDICATOR_SPECS = {
    "D1": {
        "overlay": ["ema200", "ema50"],
        "oscillator": ["adx"],
    },
    "H4": {
        "overlay": ["ema20", "ema50"],
        "oscillator": ["adx", "macd", "macd_signal", "macd_hist"],
    },
    "H1": {
        "overlay": ["ema20", "ema50"],
        "oscillator": ["rsi"],
    },
    "M15": {
        "overlay": ["ema9", "ema20", "bb_upper", "bb_lower"],
        "oscillator": ["rsi"],
    },
}


@app.get("/api/indicators")
def get_indicators(tf: str = "M15"):
    if tf not in TIMEFRAMES:
        raise HTTPException(
            400, f"Unknown timeframe '{tf}', expected one of {TIMEFRAMES}"
        )

    spec = INDICATOR_SPECS.get(tf, {"overlay": [], "oscillator": []})
    path = DATA_FEATURES / f"{MT5_SYMBOL}_{tf}_features.parquet"
    if not path.exists():
        return {"symbol": MT5_SYMBOL, "timeframe": tf, "overlay": {}, "oscillator": {}}

    df = pd.read_parquet(path)
    times = (df["datetime"].astype("int64") // 1_000).tolist()

    def series_for(cols: list[str]) -> dict:
        result = {}
        for col in cols:
            if col not in df.columns:
                continue
            valid = df[col].notna()
            result[col] = [
                {"time": t, "value": v}
                for t, v, ok in zip(times, df[col].tolist(), valid.tolist())
                if ok
            ]
        return result

    return {
        "symbol": MT5_SYMBOL,
        "timeframe": tf,
        "overlay": series_for(spec["overlay"]),
        "oscillator": series_for(spec["oscillator"]),
    }


@app.get("/api/runs")
def list_runs():
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = []
    for run_dir in RUNS_DIR.iterdir():
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            # older runs predate created_at — fall back to folder mtime so sorting still works
            created_at = meta.get("created_at") or pd.Timestamp(run_dir.stat().st_mtime, unit="s").isoformat()
            runs.append({"run_id": run_dir.name, **meta, "created_at": created_at})
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return {"runs": runs}


@app.get("/api/runs/{run_id}/equity")
def get_run_equity(run_id: str):
    path = RUNS_DIR / run_id / "equity.json"
    if not path.exists():
        raise HTTPException(404, f"No run '{run_id}'. Run env/evaluate.py first.")
    return {"run_id": run_id, "points": json.loads(path.read_text())}


@app.get("/api/runs/{run_id}/trades")
def get_run_trades(run_id: str):
    path = RUNS_DIR / run_id / "trades.json"
    if not path.exists():
        raise HTTPException(404, f"No run '{run_id}'. Run env/evaluate.py first.")
    return {"run_id": run_id, "trades": json.loads(path.read_text())}


@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(run_id: str):
    path = RUNS_DIR / run_id / "metrics.json"
    if not path.exists():
        raise HTTPException(404, f"No metrics for '{run_id}'. Re-run env/evaluate.py to generate metrics.json.")
    return {"run_id": run_id, "metrics": json.loads(path.read_text())}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
