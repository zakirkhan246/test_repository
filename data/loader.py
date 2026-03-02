"""
Load real intraday market data from CSV files.

Handles:
- Reading 1-minute OHLCV data and resampling to 5-minute candles
- Filtering to only days with actual volume data
- Producing output in the same format as generator.py for backtest compatibility
"""

import os
import pandas as pd


def load_sensex_data(
    csv_path: str = None,
    interval_minutes: int = 5,
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

    # Drop rows with zero volume — only keep days with real volume data
    df = df[df["volume"] > 0].copy()

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

    return df
