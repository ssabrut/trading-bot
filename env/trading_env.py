"""
env/trading_env.py
─────────────────────────────────────────────────────
Multi-timeframe Gymnasium environment. Steps on M15; H1/H4/D1 observation
windows only update when their last-closed bar changes (as-of, no lookahead).

Action space: Discrete(3) — Hold / Buy / Sell. One position at a time; once opened,
a trade rides to SL or TP with no agent-initiated early exit.
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
    "M15": [
        "ema9", "ema20", "ema_cross_up", "ema_cross_down", "rsi", "atr", "atr_pct",
        "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
        "session_asian", "session_london", "session_ny", "session_overlap", "hour_sin", "hour_cos",
    ],
    "H1": ["ema20", "ema50", "ema_cross_up", "ema_cross_down", "rsi", "atr", "atr_pct"],
    "H4": ["ema20", "ema50", "ema_trend", "adx", "macd", "macd_signal", "macd_hist"],
    "D1": ["ema200", "ema50", "ema_regime", "adx", "atr", "atr_pct", "kumo_thickness", "price_vs_kumo_top"],
}

HOLD, BUY, SELL = 0, 1, 2

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
RISK_PER_TRADE = 0.01  # fraction of equity risked per trade
SPREAD_DEFAULT = 0.00015  # fallback spread (price units) if raw spread col is 0/missing
EPISODE_DAYS_DEFAULT = 90
IDLE_PENALTY = 0.00002  # per-step reward while flat — removes HOLD-forever as a zero-cost local optimum
ADX_MIN_THRESHOLD = 15.0  # D1 ADX below this = weak-trend/choppy regime, ~15th pct of train — force HOLD
DOWNSIDE_PENALTY_COEF = 20.0  # convex penalty on negative-return steps (semi-variance style) — punishes large losing steps disproportionately more than typical ones


@dataclass
class Position:
    side: int  # 1 = long, -1 = short
    entry_price: float
    entry_time: np.datetime64
    size: float  # units of base currency
    sl: float
    tp: float
    mfe_price: float = 0.0  # best price reached so far (most favorable), init'd to entry_price on open


class MultiTimeframeTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        split: str = "train",
        symbol: str = "GBPUSD",
        episode_days: int = EPISODE_DAYS_DEFAULT,
        initial_balance: float = 10_000.0,
        seed: int | None = None,
        normalized_dir: Path | None = None,
        date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ):
        """
        normalized_dir: override for where <symbol>_<TF>_<split>.parquet normalized feature
            files live — defaults to data/normalized/. Used by walk-forward to point at
            fold-specific normalized data without touching the main pipeline's files.
        date_range: (start, end) — if given, bars are filtered by this range instead of the
            `split` column in data/processed/*.parquet. Used by walk-forward, where fold
            boundaries don't match the fixed train/val/test split.
        """
        super().__init__()
        assert split in ("train", "val", "test")
        self.split = split
        self.symbol = symbol
        self.episode_bars = episode_days * 96  # 96 M15 bars/day, 24h FX market
        self.initial_balance = initial_balance
        self._normalized_dir = normalized_dir or DATA_NORMALIZED
        self._date_range = date_range

        self._load_data()

        self.action_space = spaces.Discrete(3)
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
            tf: pd.read_parquet(self._normalized_dir / f"{self.symbol}_{tf}_{self.split}.parquet")
            for tf in TIMEFRAMES
        }
        # raw (unscaled) ATR and ADX — ATR needed for SL/TP distance in price units,
        # ADX needed to gate trading during weak-trend/choppy regimes (see _regime_gate_active).
        self.raw_feat = {}
        for tf in TIMEFRAMES:
            full = pd.read_parquet(DATA_FEATURES / f"{self.symbol}_{tf}_features.parquet")
            keep = ["datetime"] + [c for c in ("atr", "adx") if c in full.columns]
            if len(keep) > 1:
                self.raw_feat[tf] = full[keep]
        # raw OHLC + spread for execution, filtered to this split (or explicit date_range for walk-forward)
        proc = pd.read_parquet(DATA_PROCESSED / f"{self.symbol}_M15.parquet")
        if self._date_range is not None:
            start, end = self._date_range
            self.bars = proc[(proc["datetime"] >= start) & (proc["datetime"] < end)].reset_index(drop=True)
        else:
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

    def _raw_feat_value(self, tf: str, col: str, t: np.datetime64) -> float:
        df = self.raw_feat[tf]
        dt = df["datetime"].values
        idx = np.searchsorted(dt, t, side="right") - 1
        if idx < 0:
            idx = 0
        return float(df[col].iloc[idx])

    def _raw_atr(self, tf: str, t: np.datetime64) -> float:
        return self._raw_feat_value(tf, "atr", t)

    def _regime_gate_active(self, t: np.datetime64) -> bool:
        """True when D1 ADX is below threshold — weak-trend/choppy regime, don't force a directional bet."""
        d1_adx = self._raw_feat_value("D1", "adx", t)
        return d1_adx < ADX_MIN_THRESHOLD

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
        self._last_close_mfe_fraction: float | None = None

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
            size=size, sl=sl, tp=tp, mfe_price=price,
        )

    def _close_position(self, price: float | None = None, reason: str = "manual"):
        if self.position is None:
            return 0.0
        price = price if price is not None else self._current_price()
        pnl = self._unrealized_pnl(price)
        self.balance += pnl
        pos = self.position

        tp_dist = abs(pos.tp - pos.entry_price)
        mfe_dist = pos.side * (pos.mfe_price - pos.entry_price)  # positive = favorable
        mfe_fraction = max(0.0, mfe_dist / tp_dist) if tp_dist > 0 else 0.0
        self._last_close_mfe_fraction = mfe_fraction

        self.trade_log.append(
            {
                "entry_time": pos.entry_time,
                "entry_price": pos.entry_price,
                "exit_time": self._bars_dt[self.cursor],
                "exit_price": price,
                "side": pos.side,
                "size": pos.size,
                "sl": pos.sl,
                "tp": pos.tp,
                "pnl": pnl,
                "reason": reason,
                "mfe_fraction": round(mfe_fraction, 4),
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

        # track most-favorable-excursion using intra-bar high/low, before checking SL/TP
        if pos.side == 1:
            pos.mfe_price = max(pos.mfe_price, high)
        else:
            pos.mfe_price = min(pos.mfe_price, low)

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

        regime_gated = self.position is None and self._regime_gate_active(self._bars_dt[self.cursor])
        if regime_gated:
            action = HOLD  # weak-trend/choppy D1 regime — don't force a directional bet

        if action == BUY and self.position is None:
            self._open_position(1)
        elif action == SELL and self.position is None:
            self._open_position(-1)
        # HOLD, or BUY/SELL while a position is already open: no-op

        self.cursor += 1
        terminated = False
        truncated = self.cursor >= self.end_idx or self.cursor >= len(self.bars) - 1

        if truncated and self.position is not None:
            realized += self._close_position(self._current_price(), reason="episode_end")

        price = self._current_price()
        equity_after = self.balance + self._unrealized_pnl(price)
        self.peak_equity = max(self.peak_equity, equity_after)

        reward = (equity_after - equity_before) / self.initial_balance

        # convex penalty on losing trade CLOSES only (not every mark-to-market step) — the
        # agent can't act mid-trade (no CLOSE action, rides to SL/TP), so penalizing every
        # negative unrealized step just added noise to steps with no counterfactual action.
        # Applying it once, tied to the realized outcome of the entry decision, targets the
        # thing the agent actually controls.
        if realized < 0:
            realized_pct = realized / self.initial_balance
            reward -= DOWNSIDE_PENALTY_COEF * realized_pct**2

        if self.position is None and not regime_gated:
            reward -= IDLE_PENALTY

        obs = self._get_obs()
        info = {"balance": self.balance, "equity": equity_after, "realized_pnl": realized}
        if truncated:
            info["trade_log"] = self.trade_log

        return obs, reward, terminated, truncated, info
