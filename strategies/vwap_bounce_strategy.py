"""
VWAP Bounce Strategy — 1-Minute Timeframe.

Market:  BSE SENSEX Index — intraday only (days with real volume)

VWAP:    Intraday Volume Weighted Average Price, resets daily.

Long Setup:
  - Price pulls back to VWAP from above
  - Candle low touches/pierces VWAP but closes above it (bounce)
  - Entry at close of bounce candle
  - SL = low of the bounce candle
  - Target = last swing high before the pullback

Short Setup:
  - Price pulls back to VWAP from below
  - Candle high touches/pierces VWAP but closes below it (rejection)
  - Entry at close of bounce candle
  - SL = high of the bounce candle
  - Target = last swing low before the pullback
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute intraday VWAP, resetting daily."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    cum_tp_vol = tp_vol.groupby(df["date"]).cumsum()
    cum_vol = df["volume"].groupby(df["date"]).cumsum()

    vwap = cum_tp_vol / cum_vol
    return vwap


def find_swing_high(highs: np.ndarray, end_idx: int, lookback: int = 60) -> float:
    """Find the highest high in the lookback window before end_idx."""
    start = max(0, end_idx - lookback)
    if start >= end_idx:
        return np.nan
    return np.max(highs[start:end_idx])


def find_swing_low(lows: np.ndarray, end_idx: int, lookback: int = 60) -> float:
    """Find the lowest low in the lookback window before end_idx."""
    start = max(0, end_idx - lookback)
    if start >= end_idx:
        return np.nan
    return np.min(lows[start:end_idx])


def count_vwap_touches(highs, lows, vwap, idx, window=10):
    """Count how many of the last `window` candles straddle VWAP (high >= VWAP >= low)."""
    start = max(0, idx - window)
    count = 0
    for j in range(start, idx):
        if highs[j] >= vwap[j] and lows[j] <= vwap[j]:
            count += 1
    return count


def detect_vwap_bounce_signals(
    df: pd.DataFrame,
    lookback: int = 60,
    min_rr: float = 1.0,
    proximity_pct: float = 0.05,
    start_candle: int = 0,
    end_candle: int = 999,
    max_vol_ratio: float = 0.0,
    max_vwap_touches: int = 0,
) -> pd.DataFrame:
    """
    Detect VWAP bounce entries on 1-min candles.

    Long bounce: candle low <= VWAP, close > VWAP, prior candles were above VWAP
    Short bounce: candle high >= VWAP, close < VWAP, prior candles were below VWAP

    Parameters
    ----------
    df : DataFrame with OHLCV + date + vwap columns
    lookback : bars to search for swing high/low target
    min_rr : minimum reward:risk ratio to take a trade
    proximity_pct : how close price needs to be to VWAP (% of price) for touch
    start_candle : first candle index allowed for signals (skip early session)
    end_candle : last candle index allowed for signals (skip late session)
    max_vol_ratio : skip candles where volume > max_vol_ratio × day avg (0 = disabled)
    max_vwap_touches : skip entry if >= this many of last 10 candles straddle VWAP (0 = disabled)
    """
    df = df.copy()
    df["signal"] = 0
    df["stop_loss"] = np.nan
    df["target"] = np.nan

    for date in df["date"].unique():
        day_mask = df["date"] == date
        day_df = df.loc[day_mask]

        if len(day_df) < 30:
            continue

        vwap = day_df["vwap"].values
        highs = day_df["high"].values
        lows = day_df["low"].values
        closes = day_df["close"].values
        opens = day_df["open"].values
        volumes = day_df["volume"].values
        indices = day_df.index

        candle_idxs = day_df["candle_idx"].values

        for i in range(20, len(day_df) - 5):
            ci = candle_idxs[i]
            if ci < start_candle or ci > end_candle:
                continue

            v = vwap[i]
            if np.isnan(v) or v <= 0:
                continue

            if max_vol_ratio > 0:
                avg_vol = np.mean(volumes[:i+1])
                if avg_vol > 0 and volumes[i] / avg_vol > max_vol_ratio:
                    continue

            if max_vwap_touches > 0:
                touches = count_vwap_touches(highs, lows, vwap, i, window=10)
                if touches >= max_vwap_touches:
                    continue

            h = highs[i]
            l = lows[i]
            c = closes[i]
            o = opens[i]

            # --- LONG BOUNCE ---
            # Candle touches VWAP from above: low <= VWAP, close > VWAP
            # Candle must be bullish (close > open) to confirm direction
            if l <= v and c > v and c > o:
                # Confirm prior context: majority of last 5 closes above VWAP
                recent_above = sum(1 for j in range(i-5, i) if closes[j] > vwap[j])
                if recent_above >= 3:
                    sl = l
                    risk = c - sl
                    if risk > 0:
                        target = find_swing_high(highs, i, lookback)
                        if not np.isnan(target) and target > c:
                            reward = target - c
                            if reward / risk >= min_rr:
                                idx = indices[i]
                                df.at[idx, "signal"] = 1
                                df.at[idx, "stop_loss"] = sl
                                df.at[idx, "target"] = target

            # --- SHORT BOUNCE ---
            # Candle touches VWAP from below: high >= VWAP, close < VWAP
            # Candle must be bearish (close < open) to confirm direction
            if h >= v and c < v and c < o:
                # Confirm prior context: majority of last 5 closes below VWAP
                recent_below = sum(1 for j in range(i-5, i) if closes[j] < vwap[j])
                if recent_below >= 3:
                    sl = h
                    risk = sl - c
                    if risk > 0:
                        target = find_swing_low(lows, i, lookback)
                        if not np.isnan(target) and target < c:
                            reward = c - target
                            if reward / risk >= min_rr:
                                idx = indices[i]
                                df.at[idx, "signal"] = -1
                                df.at[idx, "stop_loss"] = sl
                                df.at[idx, "target"] = target

    return df
