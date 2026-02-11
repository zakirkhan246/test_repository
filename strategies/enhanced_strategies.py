"""
Enhanced intraday strategies with additional filters for higher win rates.

These strategies build on the base MACD/RSI/Momentum framework but add:
- EMA trend filters (only trade in trend direction)
- Volatility squeeze detection (trade breakouts from low-vol periods)
- VWAP confluence (institutional price level)
- Pullback entries (better entry prices within trends)
- Multi-timeframe confirmation via longer-period indicators
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, Signal
from .indicators import (
    compute_macd, compute_rsi, compute_momentum, compute_atr,
    compute_ema, compute_vwap, compute_bollinger_bandwidth,
)


class TrendFilteredMACD(BaseStrategy):
    """
    Strategy 6: EMA Trend-Filtered MACD Crossover

    Only takes MACD crossover signals in the direction of the prevailing
    trend defined by 50-period EMA. Also requires RSI confirmation.

    LONG: Price above EMA50 + MACD crosses up + RSI 40-65 + Momentum > 0
    SHORT: Price below EMA50 + MACD crosses down + RSI 35-60 + Momentum < 0

    The EMA filter dramatically reduces false signals by ensuring we only
    trade with the trend.
    """

    def __init__(self, ema_period=50, **kwargs):
        super().__init__(**kwargs)
        self.ema_period = ema_period

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare_indicators(df)
        df["ema_trend"] = compute_ema(df["close"], self.ema_period)
        df["ema_slope"] = df["ema_trend"] - df["ema_trend"].shift(3)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Trend filter: price above/below EMA with EMA sloping in right direction
        uptrend = (df["close"] > df["ema_trend"]) & (df["ema_slope"] > 0)
        downtrend = (df["close"] < df["ema_trend"]) & (df["ema_slope"] < 0)

        # MACD crossover
        macd_cross_up = df["macd_cross_up"]
        macd_cross_down = df["macd_cross_down"]

        # RSI in healthy zone (not overbought/oversold)
        rsi_long = (df["rsi"] > 40) & (df["rsi"] < 65)
        rsi_short = (df["rsi"] > 35) & (df["rsi"] < 60)

        # Momentum confirmation
        mom_long = df["momentum"] > 0
        mom_short = df["momentum"] < 0

        long_cond = macd_cross_up & uptrend & rsi_long & mom_long & in_window
        short_cond = macd_cross_down & downtrend & rsi_short & mom_short & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class PullbackMomentum(BaseStrategy):
    """
    Strategy 7: Pullback Entry with Momentum Confirmation

    Waits for a pullback in a trending market, then enters when momentum
    resumes in the trend direction.

    LONG: Price was above EMA20, pulled back to within 0.2% of EMA20,
          RSI dipped below 45 then recovered above 50, momentum turning up
    SHORT: Price was below EMA20, pulled back toward EMA20,
           RSI spiked above 55 then dropped below 50, momentum turning down

    This gets better entry prices by waiting for pullbacks.
    """

    def __init__(self, ema_period=20, pullback_pct=0.002, **kwargs):
        super().__init__(**kwargs)
        self.ema_period = ema_period
        self.pullback_pct = pullback_pct

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare_indicators(df)
        df["ema_fast"] = compute_ema(df["close"], self.ema_period)
        df["ema_slow"] = compute_ema(df["close"], 50)
        df["dist_to_ema"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Trend definition: EMA20 above EMA50 = uptrend
        uptrend = df["ema_fast"] > df["ema_slow"]
        downtrend = df["ema_fast"] < df["ema_slow"]

        # Pullback detection: price near EMA in trend
        near_ema_from_above = (df["dist_to_ema"] > -self.pullback_pct) & (
            df["dist_to_ema"] < self.pullback_pct
        )

        # RSI recovering from pullback
        rsi_recovering = (df["rsi"] > 48) & (df["rsi"].shift(1) < 48)
        rsi_failing = (df["rsi"] < 52) & (df["rsi"].shift(1) > 52)

        # Momentum turning
        mom_turn_up = df["mom_rising"] & (df["momentum"].shift(1) < df["momentum"])
        mom_turn_down = (~df["mom_rising"]) & (df["momentum"].shift(1) > df["momentum"])

        # MACD histogram positive/negative as extra filter
        macd_bull = df["macd_hist"] > 0
        macd_bear = df["macd_hist"] < 0

        long_cond = (
            uptrend & near_ema_from_above & rsi_recovering
            & mom_turn_up & macd_bull & in_window
        )
        short_cond = (
            downtrend & near_ema_from_above & rsi_failing
            & mom_turn_down & macd_bear & in_window
        )

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class VolatilitySqueezeBreakout(BaseStrategy):
    """
    Strategy 8: Volatility Squeeze Breakout

    Detects periods of low volatility (Bollinger Bandwidth contraction)
    and trades the breakout when confirmed by MACD + RSI + Momentum.

    Entry after Bollinger Bandwidth drops below its 20-period moving average
    then expands, combined with directional confirmation from all 3 indicators.

    Low volatility precedes large moves - this captures them.
    """

    def __init__(self, bb_period=20, squeeze_lookback=20, **kwargs):
        super().__init__(**kwargs)
        self.bb_period = bb_period
        self.squeeze_lookback = squeeze_lookback

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare_indicators(df)
        df["bb_bandwidth"] = compute_bollinger_bandwidth(df["close"], self.bb_period)
        df["bb_bw_ma"] = df["bb_bandwidth"].rolling(self.squeeze_lookback).mean()
        df["in_squeeze"] = df["bb_bandwidth"] < df["bb_bw_ma"]
        df["squeeze_release"] = (~df["in_squeeze"]) & df["in_squeeze"].shift(1)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # Squeeze release (volatility expanding from contraction)
        squeeze = df["squeeze_release"].fillna(False)

        # Direction from MACD
        macd_bull = (df["macd_line"] > df["signal_line"]) & (df["macd_hist"] > 0)
        macd_bear = (df["macd_line"] < df["signal_line"]) & (df["macd_hist"] < 0)

        # RSI direction
        rsi_bull = (df["rsi"] > 50) & (df["rsi"] < 70)
        rsi_bear = (df["rsi"] < 50) & (df["rsi"] > 30)

        # Momentum confirmation
        mom_bull = df["momentum"] > 0
        mom_bear = df["momentum"] < 0

        long_cond = squeeze & macd_bull & rsi_bull & mom_bull & in_window
        short_cond = squeeze & macd_bear & rsi_bear & mom_bear & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class EMACrossoverTriple(BaseStrategy):
    """
    Strategy 9: EMA Crossover with Triple Indicator Confirmation

    Uses fast/slow EMA crossover as primary signal with MACD histogram,
    RSI zone, and momentum all confirming the direction.

    LONG: EMA9 crosses above EMA21 + MACD histogram positive
          + RSI > 50 + Momentum positive
    SHORT: EMA9 crosses below EMA21 + MACD histogram negative
           + RSI < 50 + Momentum negative

    The EMA crossover provides cleaner trend signals than MACD alone.
    """

    def __init__(self, fast_ema=9, slow_ema=21, **kwargs):
        super().__init__(**kwargs)
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare_indicators(df)
        df["ema_fast"] = compute_ema(df["close"], self.fast_ema)
        df["ema_slow"] = compute_ema(df["close"], self.slow_ema)
        df["ema_cross_up"] = (df["ema_fast"] > df["ema_slow"]) & (
            df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)
        )
        df["ema_cross_down"] = (df["ema_fast"] < df["ema_slow"]) & (
            df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)
        )
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # EMA crossover
        ema_up = df["ema_cross_up"]
        ema_down = df["ema_cross_down"]

        # MACD histogram confirmation
        macd_bull = df["macd_hist"] > 0
        macd_bear = df["macd_hist"] < 0

        # RSI confirmation
        rsi_bull = df["rsi"] > 50
        rsi_bear = df["rsi"] < 50

        # Momentum confirmation
        mom_bull = df["momentum"] > 0
        mom_bear = df["momentum"] < 0

        long_cond = ema_up & macd_bull & rsi_bull & mom_bull & in_window
        short_cond = ema_down & macd_bear & rsi_bear & mom_bear & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df


class VWAPMomentumMACD(BaseStrategy):
    """
    Strategy 10: VWAP + Momentum + MACD Confluence

    Uses VWAP as institutional reference price. Trades when price
    crosses VWAP with momentum and MACD confirmation.

    LONG: Price crosses above VWAP + MACD histogram positive & rising
          + RSI between 45-65 + Momentum positive
    SHORT: Price crosses below VWAP + MACD histogram negative & falling
           + RSI between 35-55 + Momentum negative

    VWAP is widely followed by institutions, adding a strong support/resistance level.
    """

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare_indicators(df)
        # Compute VWAP per day
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["cum_tp_vol"] = df.groupby("date").apply(
            lambda x: (x["typical_price"] * x["volume"]).cumsum()
        ).droplevel(0)
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]

        # Price crossing VWAP
        df["above_vwap"] = df["close"] > df["vwap"]
        df["vwap_cross_up"] = df["above_vwap"] & (~df["above_vwap"].shift(1).fillna(False))
        df["vwap_cross_down"] = (~df["above_vwap"]) & df["above_vwap"].shift(1).fillna(True)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = Signal.NONE

        in_window = (df["candle_idx"] >= self.entry_start_candle) & (
            df["candle_idx"] <= self.entry_end_candle
        )

        # VWAP crossover
        vwap_up = df["vwap_cross_up"]
        vwap_down = df["vwap_cross_down"]

        # MACD confirmation (histogram positive & rising for longs)
        macd_bull = (df["macd_hist"] > 0) & df["macd_hist_rising"]
        macd_bear = (df["macd_hist"] < 0) & (~df["macd_hist_rising"])

        # RSI in trend zone
        rsi_bull = (df["rsi"] > 45) & (df["rsi"] < 65)
        rsi_bear = (df["rsi"] > 35) & (df["rsi"] < 55)

        # Momentum
        mom_bull = df["momentum"] > 0
        mom_bear = df["momentum"] < 0

        long_cond = vwap_up & macd_bull & rsi_bull & mom_bull & in_window
        short_cond = vwap_down & macd_bear & rsi_bear & mom_bear & in_window

        df.loc[long_cond, "signal"] = Signal.BUY
        df.loc[short_cond, "signal"] = Signal.SELL

        return df
