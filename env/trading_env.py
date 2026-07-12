"""
env/trading_env.py
─────────────────────────────────────────────────────
Multi-timeframe Gymnasium environment. Steps on M15; H1/H4/D1 observation
windows only update when their last-closed bar changes (as-of, no lookahead).

Action space: Discrete(4) — Hold / Buy / Sell / Close.
SL/TP are ATR-based (not agent-controlled) — v1 baseline, see notebooks/5_env.ipynb.
"""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config.settings import DATA_FEATURES, DATA_PROCESSED, ROOT

DATA_NORMALIZED = ROOT / "data" / "normalized"

TIMEFRAMES = ["M15", "H1", "H4", "D1"]
WINDOW = {"M15": 64, "H1": 24, "H4": 30, "D1": 20}
FEATURE_COLS = {
    "M15": ["ema9", "ema20", "ema_cross_up", "ema_cross_down", "rsi", "atr", "atr_pct", "bb_upper", "bb_lower", "bb_width", "bb_pct_b"],
    "H1": ["ema20", "ema50", "ema_cross_up", "ema_cross_down", "rsi", "atr", "atr_pct"],
    "H4": ["ema20", "ema50", "ema_trend", "adx", "macd", "macd_signal", "macd_hist"],
    "D1": ["ema200", "ema50", "ema_regime", "adx", "atr", "atr_pct"],
}

HOLD, BUY, SELL, CLOSE = 0, 1, 2, 3

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
RISK_PER_TRADE = 0.01  # fraction of equity risked per trade
SPREAD_DEFAULT = 0.00015  # fallback spread (price units) if raw spread col is 0/missing
EPISODE_DAYS_DEFAULT = 90


@dataclass
class Position:
    side: int  # 1 = long, -1 = short
    entry_price: float
    entry_time: np.datetime64
    size: float  # units of base currency
    sl: float
    tp: float


class MultiTimeframeTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        split: str = "train",
        symbol: str = "GBPUSD",
        episode_days: int = EPISODE_DAYS_DEFAULT,
        initial_balance: float = 10_000.0,
        seed: int | None = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        self.split = split
        self.symbol = symbol
        self.episode_bars = episode_days * 96  # 96 M15 bars/day, 24h FX market
        self.initial_balance = initial_balance

        self._load_data()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                **{
                    tf.lower(): spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=(WINDOW[tf], len(FEATURE_COLS[tf])),
                        dtype=np.float32,
                    )
                    for tf in TIMEFRAMES
                },
                "portfolio": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )

        self._np_random_seed = seed
        self.reset(seed=seed)

    def _load_data(self):
        # normalized features (agent input) — indexed by TF
        self.norm = {
            tf: pd.read_parquet(DATA_NORMALIZED / f"{self.symbol}_{tf}_{self.split}.parquet")
            for tf in TIMEFRAMES
        }
        # raw ATR (unscaled, price units) — needed for SL/TP distance
        self.raw_feat = {
            tf: pd.read_parquet(DATA_FEATURES / f"{self.symbol}_{tf}_features.parquet")[["datetime", "atr"]]
            for tf in TIMEFRAMES
            if "atr" in pd.read_parquet(DATA_FEATURES / f"{self.symbol}_{tf}_features.parquet").columns
        }
        # raw OHLC + spread for execution, filtered to this split
        proc = pd.read_parquet(DATA_PROCESSED / f"{self.symbol}_M15.parquet")
        self.bars = proc[proc["split"] == self.split].reset_index(drop=True)

        # precompute per-TF datetime numpy arrays for fast searchsorted as-of lookup
        self._dt = {tf: self.norm[tf]["datetime"].values for tf in TIMEFRAMES}
        self._feat_arr = {tf: self.norm[tf][FEATURE_COLS[tf]].values.astype(np.float32) for tf in TIMEFRAMES}
        self._bars_dt = self.bars["datetime"].values

        # indicator warmup (e.g. EMA200 on D1) leaves NaN rows at the start of each TF's
        # feature file. Episodes must not start until every TF's full lookback WINDOW is
        # past warmup — not just the as-of bar itself, or the window's earliest rows leak NaN.
        warmup_dt = max(
            self.norm[tf]["datetime"].iloc[
                self.norm[tf][FEATURE_COLS[tf]].notna().all(axis=1).idxmax() + WINDOW[tf] - 1
            ]
            for tf in TIMEFRAMES
        )
        self._warmup_bar_idx = int(np.searchsorted(self._bars_dt, pd.Timestamp(warmup_dt).to_datetime64(), side="right"))

    def _asof_index(self, tf: str, t: np.datetime64) -> int:
        """Index of the latest CLOSED bar at or before time t (no lookahead)."""
        idx = np.searchsorted(self._dt[tf], t, side="right") - 1
        return int(idx)

    def _window(self, tf: str, t: np.datetime64) -> np.ndarray:
        end = self._asof_index(tf, t)
        w = WINDOW[tf]
        start = end - w + 1
        if start < 0:
            pad = np.zeros((-start, len(FEATURE_COLS[tf])), dtype=np.float32)
            return np.concatenate([pad, self._feat_arr[tf][: end + 1]], axis=0)
        return self._feat_arr[tf][start : end + 1]

    def _raw_atr(self, tf: str, t: np.datetime64) -> float:
        df = self.raw_feat[tf]
        dt = df["datetime"].values
        idx = np.searchsorted(dt, t, side="right") - 1
        if idx < 0:
            return float(df["atr"].iloc[0])
        return float(df["atr"].iloc[idx])

    def _get_obs(self) -> dict:
        t = self._bars_dt[self.cursor]
        obs = {tf.lower(): self._window(tf, t) for tf in TIMEFRAMES}

        price = self._current_price()
        unrealized = self._unrealized_pnl(price)
        equity = self.balance + unrealized
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        obs["portfolio"] = np.array(
            [
                self.balance / self.initial_balance,
                equity / self.initial_balance,
                float(self.position.side) if self.position else 0.0,
                unrealized / self.initial_balance,
                drawdown,
                self.bars["spread"].iloc[self.cursor] or SPREAD_DEFAULT,
            ],
            dtype=np.float32,
        )
        return obs

    def _current_price(self) -> float:
        return float(self.bars["close"].iloc[self.cursor])

    def _unrealized_pnl(self, price: float) -> float:
        if self.position is None:
            return 0.0
        return self.position.side * (price - self.position.entry_price) * self.position.size

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        max_start = len(self.bars) - self.episode_bars - 1
        min_start = self._warmup_bar_idx  # skip until all TFs' indicators are past warmup (no NaNs)
        if max_start <= min_start:
            self.start_idx = 0
            self.episode_bars = len(self.bars) - 1
        else:
            self.start_idx = int(self.np_random.integers(min_start, max_start))

        self.cursor = self.start_idx
        self.end_idx = self.start_idx + self.episode_bars

        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position: Position | None = None
        self.trade_log: list[dict] = []

        obs = self._get_obs()
        info = {}
        return obs, info

    def _open_position(self, side: int):
        price = self._current_price()
        atr = self._raw_atr("M15", self._bars_dt[self.cursor])
        sl_dist = SL_ATR_MULT * atr
        tp_dist = TP_ATR_MULT * atr
        if sl_dist <= 0:
            return

        risk_amount = self.balance * RISK_PER_TRADE
        size = risk_amount / sl_dist

        sl = price - side * sl_dist
        tp = price + side * tp_dist
        self.position = Position(
            side=side, entry_price=price, entry_time=self._bars_dt[self.cursor],
            size=size, sl=sl, tp=tp,
        )

    def _close_position(self, price: float | None = None, reason: str = "manual"):
        if self.position is None:
            return 0.0
        price = price if price is not None else self._current_price()
        pnl = self._unrealized_pnl(price)
        self.balance += pnl
        pos = self.position
        self.trade_log.append(
            {
                "entry_time": pos.entry_time,
                "entry_price": pos.entry_price,
                "exit_time": self._bars_dt[self.cursor],
                "exit_price": price,
                "side": pos.side,
                "size": pos.size,
                "pnl": pnl,
                "reason": reason,
            }
        )
        self.position = None
        return pnl

    def _check_sl_tp(self) -> float:
        """Check if current bar's high/low crossed SL/TP; close at that price if so."""
        if self.position is None:
            return 0.0
        high = float(self.bars["high"].iloc[self.cursor])
        low = float(self.bars["low"].iloc[self.cursor])
        pos = self.position

        if pos.side == 1:
            hit_sl = low <= pos.sl
            hit_tp = high >= pos.tp
        else:
            hit_sl = high >= pos.sl
            hit_tp = low <= pos.tp

        if hit_sl and hit_tp:
            return self._close_position(pos.sl, reason="sl")  # conservative: assume SL hit first
        if hit_sl:
            return self._close_position(pos.sl, reason="sl")
        if hit_tp:
            return self._close_position(pos.tp, reason="tp")
        return 0.0

    def step(self, action: int):
        equity_before = self.balance + self._unrealized_pnl(self._current_price())

        realized = self._check_sl_tp()

        if action == BUY and self.position is None:
            self._open_position(1)
        elif action == SELL and self.position is None:
            self._open_position(-1)
        elif action == CLOSE and self.position is not None:
            realized += self._close_position(reason="agent")
        # HOLD: no-op

        self.cursor += 1
        terminated = False
        truncated = self.cursor >= self.end_idx or self.cursor >= len(self.bars) - 1

        if truncated and self.position is not None:
            realized += self._close_position(self._current_price(), reason="episode_end")

        price = self._current_price()
        equity_after = self.balance + self._unrealized_pnl(price)
        self.peak_equity = max(self.peak_equity, equity_after)

        reward = (equity_after - equity_before) / self.initial_balance

        obs = self._get_obs()
        info = {"balance": self.balance, "equity": equity_after, "realized_pnl": realized}
        if truncated:
            info["trade_log"] = self.trade_log

        return obs, reward, terminated, truncated, info
