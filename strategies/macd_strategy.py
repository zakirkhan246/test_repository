"""
Pure MACD Crossover Strategy for intraday trading.

Entry signals:
  LONG:  MACD line crosses above signal line + histogram accelerating up
         (first positive histogram bar is growing)
  SHORT: MACD line crosses below signal line + histogram accelerating down
         (first negative histogram bar is growing in magnitude)

The histogram acceleration filter rejects weak crossovers where MACD barely
nudges past the signal line before dying out.

Exit: ATR-based stop loss with 1:2 risk-reward target.
"""

import pandas as pd
from .base_strategy import BaseStrategy, Signal


class MACDCrossover(BaseStrategy):

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Core signal: MACD line crossing signal line
        cross_up = df["macd_cross_up"]
        cross_down = df["macd_cross_down"]

        # Histogram acceleration: confirms the crossover has momentum behind it
        # For longs: histogram just turned positive AND is growing
        # For shorts: histogram just turned negative AND magnitude is growing
        hist_accel_up = (df["macd_hist"] > 0) & df["macd_hist_rising"]
        hist_accel_down = (df["macd_hist"] < 0) & (~df["macd_hist_rising"])

        long_cond = cross_up & hist_accel_up & in_window
        short_cond = cross_down & hist_accel_down & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df
