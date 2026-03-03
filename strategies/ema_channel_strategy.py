"""
20 EMA Channel Strategy for intraday trading.

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
"""

import pandas as pd
from .indicators import compute_ema, compute_atr


class EMAChannelStrategy:
    """
    Computes 20 EMA channel indicators for use with ChannelEngine.

    The channel is formed by two 20-period EMAs:
    - EMA of open prices
    - EMA of close prices
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 14):
        self.ema_period = ema_period
        self.atr_period = atr_period

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EMA channel indicators and return enriched DataFrame."""
        df = df.copy()

        # 20 EMA on open and close prices
        df["ema_open"] = compute_ema(df["open"], self.ema_period)
        df["ema_close"] = compute_ema(df["close"], self.ema_period)

        # Channel boundaries (upper/lower swap depending on which EMA is higher)
        df["channel_upper"] = df[["ema_open", "ema_close"]].max(axis=1)
        df["channel_lower"] = df[["ema_open", "ema_close"]].min(axis=1)

        # ATR for reference
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.atr_period)

        # Candle index within day
        df["candle_idx"] = df.groupby("date").cumcount()

        return df
