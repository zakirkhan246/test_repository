"""
Technical indicator calculations: MACD, ATR, RSI, ADX, Choppiness Index.
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


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """
    Compute ADX (Average Directional Index).

    ADX < 20-25 = weak/no trend (choppy), ADX > 25 = trending.

    Returns DataFrame with columns: plus_di, minus_di, adx
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed with Wilder's method (EMA with alpha=1/period)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1.0 / period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / atr.replace(0, np.nan)
    minus_di = 100 * smooth_minus_dm / atr.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx},
        index=high.index,
    )


def compute_choppiness_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Compute Choppiness Index (CI).

    CI = 100 * LOG10(SUM(ATR, period) / (highest_high - lowest_low)) / LOG10(period)

    CI > 61.8 = choppy/ranging market
    CI < 38.2 = trending market
    Values between = transitional
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = true_range.rolling(window=period).sum()
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    hl_range = (highest_high - lowest_low).replace(0, np.nan)

    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(period)

    return ci
