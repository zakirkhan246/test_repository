"""
Technical indicator calculations: MACD, RSI, and Momentum.

All functions operate on pandas Series (typically 'close' prices)
and return pandas Series/DataFrames aligned to the input index.
"""

import pandas as pd
import numpy as np


def compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Returns DataFrame with columns: macd_line, signal_line, histogram
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram},
        index=close.index,
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index) using Wilder's smoothing.

    Returns Series with RSI values (0-100).
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)  # Neutral when undefined

    return rsi


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Compute Rate of Change (ROC) momentum.

    Returns percentage change over `period` bars.
    """
    momentum = ((close - close.shift(period)) / close.shift(period)) * 100.0
    return momentum


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Compute Average True Range for stop-loss/target calculation.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()

    return atr


def compute_ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series) -> pd.Series:
    """Compute Volume Weighted Average Price (cumulative within groups)."""
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def compute_bollinger_bandwidth(close: pd.Series, period: int = 20) -> pd.Series:
    """Compute Bollinger Bandwidth (volatility squeeze indicator)."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bandwidth = ((upper - lower) / sma) * 100
    return bandwidth
