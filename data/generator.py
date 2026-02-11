"""
Realistic intraday data generator for NIFTY 50 and SENSEX.

Generates synthetic OHLCV data that mimics real Indian market behavior:
- 9:15 AM to 3:30 PM IST trading hours
- 5-minute candle intervals (75 candles per day)
- Realistic volatility, gap opens, and intraday patterns
- Mean-reverting tendencies with trending periods
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_intraday_data(
    symbol: str = "NIFTY50",
    num_days: int = 250,
    interval_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic intraday OHLCV data for Indian indices."""
    rng = np.random.RandomState(seed)

    if symbol == "NIFTY50":
        base_price = 22000.0
        daily_vol = 0.012  # ~1.2% daily volatility
        tick_size = 0.05
    elif symbol == "SENSEX":
        base_price = 72000.0
        daily_vol = 0.011
        tick_size = 0.05
    else:
        raise ValueError(f"Unknown symbol: {symbol}")

    candles_per_day = (6 * 60 + 15) // interval_minutes  # 9:15 to 15:30

    # Intraday volatility profile (U-shaped: higher at open/close)
    x = np.linspace(0, 1, candles_per_day)
    vol_profile = 1.0 + 0.8 * (4 * (x - 0.5) ** 2)
    vol_profile /= vol_profile.mean()

    all_rows = []
    current_price = base_price

    # Generate trading days (skip weekends)
    start_date = datetime(2024, 1, 2)
    trading_days = []
    d = start_date
    while len(trading_days) < num_days:
        if d.weekday() < 5:  # Mon-Fri
            trading_days.append(d)
        d += timedelta(days=1)

    for day in trading_days:
        # Gap open: overnight move
        gap = rng.normal(0, daily_vol * 0.3) * current_price
        current_price += gap

        # Daily drift (slight upward bias for Indian markets)
        daily_drift = rng.normal(0.0002, daily_vol) * current_price
        drift_per_candle = daily_drift / candles_per_day

        # Intraday regime: trending (30%) or mean-reverting (70%)
        trending = rng.random() < 0.30
        if trending:
            trend_dir = rng.choice([-1, 1])
            trend_strength = rng.uniform(0.5, 1.5)
        else:
            trend_dir = 0
            trend_strength = 0

        intraday_prices = [current_price]

        for i in range(candles_per_day):
            candle_vol = daily_vol / np.sqrt(candles_per_day) * vol_profile[i]

            # Random walk component
            noise = rng.normal(0, candle_vol * current_price)

            # Mean reversion to day open
            mr_force = -0.02 * (current_price - intraday_prices[0])

            # Trend component
            trend_component = (
                trend_dir * trend_strength * candle_vol * current_price * 0.3
            )

            move = noise + mr_force + drift_per_candle + trend_component
            open_price = current_price

            # Generate realistic OHLC from the move
            intra_vol = abs(move) * rng.uniform(0.5, 2.0)
            if move >= 0:
                low = open_price - abs(rng.normal(0, intra_vol * 0.5))
                high = open_price + abs(rng.normal(0, intra_vol * 1.2))
                close = open_price + move
            else:
                high = open_price + abs(rng.normal(0, intra_vol * 0.5))
                low = open_price - abs(rng.normal(0, intra_vol * 1.2))
                close = open_price + move

            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Round to tick size
            open_price = round(open_price / tick_size) * tick_size
            high = round(high / tick_size) * tick_size
            low = round(low / tick_size) * tick_size
            close = round(close / tick_size) * tick_size

            # Volume (higher at open/close, lower midday)
            base_volume = 50000 if symbol == "NIFTY50" else 30000
            volume = int(base_volume * vol_profile[i] * rng.uniform(0.5, 1.5))

            timestamp = day.replace(hour=9, minute=15) + timedelta(
                minutes=i * interval_minutes
            )

            all_rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

            current_price = close
            intraday_prices.append(close)

    df = pd.DataFrame(all_rows)
    df["symbol"] = symbol
    df["date"] = df["timestamp"].dt.date
    df.set_index("timestamp", inplace=True)
    return df
