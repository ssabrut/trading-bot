from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── MT5 ────────────────────────────────────────────────────────────────────────
MT5_SYMBOL = "GBPUSD"
MT5_TF_M15 = 15  # minutes — maps to mt5.TIMEFRAME_M15
MT5_TF_H1 = 60  # context timeframe
MT5_TF_H4 = 240  # context timeframe
MT5_TF_D1 = 1440  # context timeframe

# ── Instruments (Dukascopy symbol per tradable instrument) ─────────────────────
INSTRUMENTS = {
    "GBPUSD": "GBP/USD",
    "NAS100": "E_NQ-100",
}
