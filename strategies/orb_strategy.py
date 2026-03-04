"""
Opening Range Breakout (ORB) Strategy.

Opening Range: High/Low of the first N candles (default 6 = 30 min on 5-min TF).

Entry signals:
  LONG:  Close breaks above OR high (after the opening range is set)
  SHORT: Close breaks below OR low (after the opening range is set)

Only one entry per direction per day — once a breakout fires, that side is done.
Re-entry allowed if a breakout in the opposite direction triggers.

Risk management:
  Stop Loss: OR low for longs, OR high for shorts (the opposite boundary)
  Target:    1:2 risk-reward from entry price

Trend Filter (same 3-factor composite as EMA channel):
  VWAP + Opening Range position + EMA(20) slope → score -3 to +3
  Score >= threshold → allow longs
  Score <= -threshold → allow shorts
"""

import pandas as pd
import numpy as np
from .indicators import compute_ema, compute_atr, compute_vwap


class ORBStrategy:
    """
    Computes Opening Range Breakout indicators and trend score
    for use with BacktestEngine.
    """

    def __init__(
        self,
        or_candles: int = 6,       # 6 × 5min = 30 min opening range
        ema_period: int = 20,
        atr_period: int = 14,
        slope_lookback: int = 3,
        rr_ratio: float = 2.0,
        entry_start_candle: int = 6,   # First candle after OR
        entry_end_candle: int = 65,    # No new entries after 14:40
    ):
        self.or_candles = or_candles
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.slope_lookback = slope_lookback
        self.rr_ratio = rr_ratio
        self.entry_start_candle = entry_start_candle
        self.entry_end_candle = entry_end_candle

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ORB levels, breakout signals, SL/target, and trend score."""
        df = df.copy()

        # Candle index within day
        df["candle_idx"] = df.groupby("date").cumcount()

        # ── Opening Range (first N candles per day) ──
        or_data = (
            df[df["candle_idx"] < self.or_candles]
            .groupby("date")
            .agg(or_high=("high", "max"), or_low=("low", "min"))
        )
        df["or_high"] = df["date"].map(or_data["or_high"])
        df["or_low"] = df["date"].map(or_data["or_low"])
        df["or_range"] = df["or_high"] - df["or_low"]

        # ATR for reference
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.atr_period)

        # ── Trend Score (same 3-factor composite as EMA channel) ──
        # Factor 1: VWAP
        df["vwap"] = compute_vwap(
            df["high"], df["low"], df["close"], df["volume"], df["date"]
        )

        # Factor 2: Close vs Opening Range
        or_signal = np.where(
            df["close"] > df["or_high"], 1,
            np.where(df["close"] < df["or_low"], -1, 0),
        )

        # Factor 3: EMA(20) slope
        ema20 = compute_ema(df["close"], self.ema_period)
        ema_slope = ema20 - ema20.shift(self.slope_lookback)
        slope_signal = np.where(
            ema_slope > 0, 1,
            np.where(ema_slope < 0, -1, 0),
        )

        vwap_signal = np.where(
            df["close"] > df["vwap"], 1,
            np.where(df["close"] < df["vwap"], -1, 0),
        )

        df["trend_score"] = vwap_signal + or_signal + slope_signal

        # ── Breakout Signals ──
        # Signal = 0 (no signal), 1 (long breakout), -1 (short breakout)
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["target"] = np.nan

        for date in df["date"].unique():
            day_mask = df["date"] == date
            day_df = df[day_mask]

            long_fired = False
            short_fired = False

            for idx in day_df.index:
                row = df.loc[idx]
                ci = row["candle_idx"]

                # Only signal after opening range and within trading window
                if ci < self.entry_start_candle or ci > self.entry_end_candle:
                    continue

                close = row["close"]
                or_high = row["or_high"]
                or_low = row["or_low"]

                # Long breakout: close above OR high
                if not long_fired and close > or_high:
                    risk = close - or_low
                    if risk > 0:
                        df.at[idx, "signal"] = 1
                        df.at[idx, "stop_loss"] = or_low
                        df.at[idx, "target"] = close + risk * self.rr_ratio
                        long_fired = True

                # Short breakout: close below OR low
                elif not short_fired and close < or_low:
                    risk = or_high - close
                    if risk > 0:
                        df.at[idx, "signal"] = -1
                        df.at[idx, "stop_loss"] = or_high
                        df.at[idx, "target"] = close - risk * self.rr_ratio
                        short_fired = True

        return df
