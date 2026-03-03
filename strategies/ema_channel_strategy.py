"""
20 EMA Channel Strategy with Multi-Factor Trend Filter.

Channel formation:
  Upper: max(EMA_20(open), EMA_20(close))
  Lower: min(EMA_20(open), EMA_20(close))

Entry signals:
  LONG:  Price closes above channel, having previously been below it
  SHORT: Price closes below channel, having previously been above it

Exit:
  LONG exit:  Price closes below channel
  SHORT exit: Price closes above channel

Re-entry (opposite direction):
  After exiting long, if the very next candle also closes below channel → SHORT
  After exiting short, if the very next candle also closes above channel → LONG

Trend Filter (3-factor composite score):
  1. VWAP: close vs intraday VWAP → +1 (above) / -1 (below)
  2. Opening Range: close vs first-30-min high/low → +1 / -1 / 0
  3. EMA Channel Slope: channel direction → +1 (rising) / -1 (falling)

  Score >= +2 → BULLISH (only longs)
  Score <= -2 → BEARISH (only shorts)
  Otherwise  → NEUTRAL (no entries)
"""

import pandas as pd
import numpy as np
from .indicators import compute_ema, compute_atr, compute_vwap


class EMAChannelStrategy:
    """
    Computes 20 EMA channel indicators and multi-factor trend score
    for use with ChannelEngine.
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        slope_lookback: int = 3,
        or_candles: int = 6,
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.slope_lookback = slope_lookback
        self.or_candles = or_candles

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA channel indicators, VWAP, OR, and trend score."""
        df = df.copy()

        # ── 20 EMA Channel ──
        df["ema_open"] = compute_ema(df["open"], self.ema_period)
        df["ema_close"] = compute_ema(df["close"], self.ema_period)
        df["channel_upper"] = df[["ema_open", "ema_close"]].max(axis=1)
        df["channel_lower"] = df[["ema_open", "ema_close"]].min(axis=1)

        # ATR for reference
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.atr_period)

        # Candle index within day
        df["candle_idx"] = df.groupby("date").cumcount()

        # ── Factor 1: VWAP (resets daily) ──
        df["vwap"] = compute_vwap(
            df["high"], df["low"], df["close"], df["volume"], df["date"]
        )

        # ── Factor 2: Opening Range (first N candles per day) ──
        or_data = (
            df[df["candle_idx"] < self.or_candles]
            .groupby("date")
            .agg(or_high=("high", "max"), or_low=("low", "min"))
        )
        df["or_high"] = df["date"].map(or_data["or_high"])
        df["or_low"] = df["date"].map(or_data["or_low"])

        # ── Factor 3: EMA Channel Slope ──
        channel_mid = (df["channel_upper"] + df["channel_lower"]) / 2
        df["channel_slope"] = channel_mid - channel_mid.shift(self.slope_lookback)

        # ── Composite Trend Score ──
        vwap_signal = np.where(
            df["close"] > df["vwap"], 1,
            np.where(df["close"] < df["vwap"], -1, 0),
        )
        or_signal = np.where(
            df["close"] > df["or_high"], 1,
            np.where(df["close"] < df["or_low"], -1, 0),
        )
        slope_signal = np.where(
            df["channel_slope"] > 0, 1,
            np.where(df["channel_slope"] < 0, -1, 0),
        )

        df["trend_score"] = vwap_signal + or_signal + slope_signal

        return df
