"""
Technical indicator calculations: MACD, ATR, RSI.
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


def compute_ema(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Exponential Moving Average.
    """
    return series.ewm(span=period, adjust=False).mean()


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    date: pd.Series,
) -> pd.Series:
    """
    Compute intraday VWAP (Volume Weighted Average Price), resetting daily.

    VWAP = cumulative(typical_price × volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    cum_tp_vol = tp_vol.groupby(date).cumsum()
    cum_vol = volume.groupby(date).cumsum()

    vwap = (cum_tp_vol / cum_vol).ffill()
    return vwap


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index).

    RSI = 100 - (100 / (1 + RS))
    where RS = EMA(gains) / EMA(losses) over `period` bars.
    """
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
