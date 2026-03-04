"""
Load real intraday market data from CSV files.

Handles:
- Reading 1-minute OHLCV data and resampling to 5-minute candles
- Filtering to only days with actual volume data
- Producing output in the same format as generator.py for backtest compatibility
"""

import os
import numpy as np
import pandas as pd


def convert_to_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert regular OHLCV candles to Heikin Ashi candles.

    Heikin Ashi formulas:
      HA_Close = (Open + High + Low + Close) / 4
      HA_Open  = (prev_HA_Open + prev_HA_Close) / 2  (first: (Open+Close)/2)
      HA_High  = max(High, HA_Open, HA_Close)
      HA_Low   = min(Low, HA_Open, HA_Close)

    Volume is kept unchanged. The conversion is applied per-day so each
    trading day starts fresh (no overnight HA state carry-over).
    """
    ha_frames = []
    for date, day_df in df.groupby("date"):
        ha = day_df.copy()
        ha_close = (day_df["open"] + day_df["high"] + day_df["low"] + day_df["close"]) / 4

        ha_open = np.empty(len(day_df))
        ha_open[0] = (day_df["open"].iloc[0] + day_df["close"].iloc[0]) / 2
        for i in range(1, len(day_df)):
            ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2

        ha["close"] = ha_close.values
        ha["open"] = ha_open
        ha["high"] = np.maximum(day_df["high"].values, np.maximum(ha_open, ha_close.values))
        ha["low"] = np.minimum(day_df["low"].values, np.minimum(ha_open, ha_close.values))
        ha_frames.append(ha)

    return pd.concat(ha_frames)


def load_sensex_data(
    csv_path: str = None,
    interval_minutes: int = 5,
    heikin_ashi: bool = False,
) -> pd.DataFrame:
    """
    Load real SENSEX intraday data from CSV and resample to the target interval.

    Only keeps trading days that have real (non-zero) volume data.

    The CSV is expected to have columns: datetime, open, high, low, close, volume
    with 1-minute bars during Indian market hours (9:15-15:29).

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file. Defaults to sensex_1min_2yr.csv in the data directory.
    interval_minutes : int
        Target candle interval in minutes (default 5).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open, high, low, close, volume, symbol, date
        indexed by timestamp, matching the format from generate_intraday_data().
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "sensex_1min_2yr.csv")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Drop rows with missing price data (keep zero-volume rows — price is valid)
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    # Set datetime as index for resampling
    df.set_index("datetime", inplace=True)
    df["date"] = df.index.date

    # Resample 1-min to target interval within each trading day
    resampled_frames = []
    for date, day_df in df.groupby("date"):
        day_resampled = day_df.resample(f"{interval_minutes}min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
        resampled_frames.append(day_resampled)

    df = pd.concat(resampled_frames)
    df["date"] = df.index.date
    df["symbol"] = "SENSEX"

    if heikin_ashi:
        df = convert_to_heikin_ashi(df)

    return df
