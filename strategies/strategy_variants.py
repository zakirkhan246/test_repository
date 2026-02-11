"""
Five distinct intraday trading strategy variants using MACD, RSI, and Momentum.

All strategies enforce:
- 1:2 risk-reward ratio
- ATR-based position sizing
- Intraday only (no overnight holds)
- Entry window: after first 30 min, before last 50 min
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal


class MACDCrossoverRSIFilter(BaseStrategy):
    """
    Strategy 1: MACD Crossover with RSI Filter

    Entry Rules:
    - LONG: MACD crosses above signal line AND RSI > 40 and < 70
      (confirms upward momentum without being overbought)
    - SHORT: MACD crosses below signal line AND RSI < 60 and > 30
      (confirms downward momentum without being oversold)

    The RSI filter prevents entries at extremes where reversals are likely.
    """

    def __init__(self, rsi_long_min=40, rsi_long_max=70, rsi_short_min=30,
                 rsi_short_max=60, **kwargs):
        super().__init__(**kwargs)
        self.rsi_long_min = rsi_long_min
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.rsi_short_max = rsi_short_max

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Long: MACD crosses up + RSI in sweet spot
        long_cond = (
            df["macd_cross_up"]
            & (df["rsi"] > self.rsi_long_min)
            & (df["rsi"] < self.rsi_long_max)
            & in_window
        )

        # Short: MACD crosses down + RSI in sweet spot
        short_cond = (
            df["macd_cross_down"]
            & (df["rsi"] < self.rsi_short_max)
            & (df["rsi"] > self.rsi_short_min)
            & in_window
        )

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class RSIReversalMomentum(BaseStrategy):
    """
    Strategy 2: RSI Reversal with Momentum Confirmation

    Entry Rules:
    - LONG: RSI crosses above oversold threshold (30->above) AND
      momentum turns positive (crossing above 0)
    - SHORT: RSI crosses below overbought threshold (70->below) AND
      momentum turns negative (crossing below 0)

    This catches mean-reversion moves confirmed by momentum shift.
    """

    def __init__(self, rsi_oversold=30, rsi_overbought=70, **kwargs):
        super().__init__(**kwargs)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # RSI crossing out of oversold
        rsi_exit_oversold = (df["rsi"] > self.rsi_oversold) & (
            df["rsi"].shift(1) <= self.rsi_oversold
        )
        # RSI crossing into overbought from above
        rsi_exit_overbought = (df["rsi"] < self.rsi_overbought) & (
            df["rsi"].shift(1) >= self.rsi_overbought
        )

        # Momentum confirmation
        mom_turning_up = (df["momentum"] > 0) | (df["mom_rising"])
        mom_turning_down = (df["momentum"] < 0) | (~df["mom_rising"])

        # Combined signals
        long_cond = rsi_exit_oversold & mom_turning_up & in_window
        short_cond = rsi_exit_overbought & mom_turning_down & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class TripleConfluence(BaseStrategy):
    """
    Strategy 3: Triple Confluence (MACD + RSI + Momentum all agree)

    Entry Rules:
    - LONG: MACD histogram positive AND rising
            + RSI between 45-65 (healthy uptrend zone)
            + Momentum positive and rising
    - SHORT: MACD histogram negative AND falling
             + RSI between 35-55 (healthy downtrend zone)
             + Momentum negative and falling

    Most conservative strategy - requires all three indicators aligned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # MACD conditions
        macd_bullish = (df["macd_hist"] > 0) & df["macd_hist_rising"]
        macd_bearish = (df["macd_hist"] < 0) & (~df["macd_hist_rising"])

        # RSI conditions (trend-following zones)
        rsi_bullish = (df["rsi"] > 45) & (df["rsi"] < 65)
        rsi_bearish = (df["rsi"] > 35) & (df["rsi"] < 55)

        # Momentum conditions
        mom_bullish = df["mom_positive"] & df["mom_rising"]
        mom_bearish = (~df["mom_positive"]) & (~df["mom_rising"])

        # Triple confluence - need fresh signal (wasn't true on prev bar)
        long_raw = macd_bullish & rsi_bullish & mom_bullish & in_window
        short_raw = macd_bearish & rsi_bearish & mom_bearish & in_window

        # Only trigger on first bar of confluence
        long_cond = long_raw & (~long_raw.shift(1).fillna(False))
        short_cond = short_raw & (~short_raw.shift(1).fillna(False))

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class MomentumBreakoutMACD(BaseStrategy):
    """
    Strategy 4: Momentum Breakout with MACD Trend Filter

    Entry Rules:
    - LONG: Momentum surges above +0.3% threshold AND
            MACD line is above signal line (uptrend confirmed)
            + Previous candle momentum was below threshold (fresh breakout)
    - SHORT: Momentum drops below -0.3% threshold AND
             MACD line is below signal line (downtrend confirmed)
             + Previous candle momentum was above threshold (fresh breakdown)

    Captures strong momentum moves with trend confirmation.
    """

    def __init__(self, momentum_threshold=0.3, **kwargs):
        super().__init__(**kwargs)
        self.momentum_threshold = momentum_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Momentum breakout (fresh)
        mom_breakout_up = (df["momentum"] > self.momentum_threshold) & (
            df["momentum"].shift(1) <= self.momentum_threshold
        )
        mom_breakout_down = (df["momentum"] < -self.momentum_threshold) & (
            df["momentum"].shift(1) >= -self.momentum_threshold
        )

        # MACD trend filter
        macd_uptrend = df["macd_line"] > df["signal_line"]
        macd_downtrend = df["macd_line"] < df["signal_line"]

        long_cond = mom_breakout_up & macd_uptrend & in_window
        short_cond = mom_breakout_down & macd_downtrend & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class AdaptiveRSIMACD(BaseStrategy):
    """
    Strategy 5: Adaptive RSI-MACD Strategy

    Uses MACD histogram strength to dynamically adjust RSI thresholds.
    When MACD shows strong momentum, RSI thresholds are relaxed.
    When MACD is weak, stricter RSI conditions are required.

    Entry Rules:
    - Compute MACD histogram percentile (strength)
    - Strong MACD (>60th pctl): RSI threshold relaxed (long >40, short <60)
    - Weak MACD (<40th pctl): RSI threshold strict (long >50, short <50)
    - LONG: Adaptive RSI condition met + MACD histogram positive + momentum rising
    - SHORT: Adaptive RSI condition met + MACD histogram negative + momentum falling
    """

    def __init__(self, lookback=50, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Rolling MACD histogram strength (percentile rank)
        hist_abs = df["macd_hist"].abs()
        df["macd_strength"] = hist_abs.rolling(self.lookback, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Adaptive RSI thresholds
        strong_macd = df["macd_strength"] > 0.6

        # Long conditions
        rsi_long_relaxed = df["rsi"] > 40
        rsi_long_strict = df["rsi"] > 50
        rsi_long_ok = (strong_macd & rsi_long_relaxed) | (~strong_macd & rsi_long_strict)

        # Short conditions
        rsi_short_relaxed = df["rsi"] < 60
        rsi_short_strict = df["rsi"] < 50
        rsi_short_ok = (strong_macd & rsi_short_relaxed) | (
            ~strong_macd & rsi_short_strict
        )

        # Combined with MACD direction and momentum
        long_raw = (
            (df["macd_hist"] > 0)
            & rsi_long_ok
            & df["mom_rising"]
            & in_window
        )
        short_raw = (
            (df["macd_hist"] < 0)
            & rsi_short_ok
            & (~df["mom_rising"])
            & in_window
        )

        # Fresh signals only
        long_cond = long_raw & (~long_raw.shift(1).fillna(False))
        short_cond = short_raw & (~short_raw.shift(1).fillna(False))

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df
