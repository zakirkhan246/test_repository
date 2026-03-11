"""
Candle Range Theory (CRT) Strategy.

CRT is a Smart Money / ICT reversal strategy based on false breakouts
(liquidity sweeps) of the previous day's range.

Range: Previous day's high and low (the "dealing range").

Sweep Detection:
  HIGH SWEEP: Candle wick pierces above prev_day_high (liquidity grab)
  LOW SWEEP:  Candle wick pierces below prev_day_low (liquidity grab)

Confirmation & Entry:
  After high sweep → candle closes back BELOW prev_day_high → SHORT
  After low sweep  → candle closes back ABOVE prev_day_low  → LONG

Risk Management:
  SHORT: SL = sweep high (max high during sweep), Target = prev_day_low
  LONG:  SL = sweep low (min low during sweep),  Target = prev_day_high

Max 1 trade per direction per day.
"""

import pandas as pd
import numpy as np
from .indicators import compute_ema, compute_atr, compute_vwap


class CRTStrategy:
    """
    Computes CRT (Candle Range Theory) signals using previous day's
    range as the dealing range and detecting liquidity sweeps.
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        entry_start_candle: int = 0,   # CRT can trigger from the open
        entry_end_candle: int = 65,    # No new entries after 14:40
    ):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.entry_start_candle = entry_start_candle
        self.entry_end_candle = entry_end_candle

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute previous day's range, detect sweeps, generate signals."""
        df = df.copy()

        # Candle index within day
        df["candle_idx"] = df.groupby("date").cumcount()

        # ── Previous Day's Range ──
        daily_hl = df.groupby("date").agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
        )
        daily_hl["prev_day_high"] = daily_hl["day_high"].shift(1)
        daily_hl["prev_day_low"] = daily_hl["day_low"].shift(1)

        df["prev_day_high"] = df["date"].map(daily_hl["prev_day_high"])
        df["prev_day_low"] = df["date"].map(daily_hl["prev_day_low"])
        df["prev_day_range"] = df["prev_day_high"] - df["prev_day_low"]
        df["prev_day_eq"] = (df["prev_day_high"] + df["prev_day_low"]) / 2

        # ATR for reference
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.atr_period)

        # ── Sweep Detection & Signal Generation ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["target"] = np.nan

        for date in df["date"].unique():
            day_mask = df["date"] == date
            day_df = df[day_mask]

            # Skip first day (no previous day data)
            if pd.isna(day_df["prev_day_high"].iloc[0]):
                continue

            pdh = day_df["prev_day_high"].iloc[0]
            pdl = day_df["prev_day_low"].iloc[0]

            # State tracking
            high_swept = False
            low_swept = False
            short_fired = False
            long_fired = False
            sweep_high_price = np.nan
            sweep_low_price = np.nan

            for idx in day_df.index:
                row = df.loc[idx]
                ci = row["candle_idx"]

                # Only signal within trading window
                if ci < self.entry_start_candle or ci > self.entry_end_candle:
                    continue

                high = row["high"]
                low = row["low"]
                close = row["close"]

                # ── HIGH SWEEP (bearish setup) ──
                if not short_fired:
                    if not high_swept and high > pdh:
                        # Sweep detected — wick pierced above prev day high
                        high_swept = True
                        sweep_high_price = high

                        # Check if same candle confirms (closes back below)
                        if close < pdh:
                            risk = sweep_high_price - close
                            if risk > 0:
                                df.at[idx, "signal"] = -1
                                df.at[idx, "stop_loss"] = sweep_high_price
                                df.at[idx, "target"] = pdl
                                short_fired = True

                    elif high_swept:
                        # Update sweep extreme if price pushes higher
                        if high > sweep_high_price:
                            sweep_high_price = high

                        # Confirmation: closes back below prev day high
                        if close < pdh:
                            risk = sweep_high_price - close
                            if risk > 0:
                                df.at[idx, "signal"] = -1
                                df.at[idx, "stop_loss"] = sweep_high_price
                                df.at[idx, "target"] = pdl
                                short_fired = True

                # ── LOW SWEEP (bullish setup) ──
                if not long_fired:
                    if not low_swept and low < pdl:
                        # Sweep detected — wick pierced below prev day low
                        low_swept = True
                        sweep_low_price = low

                        # Check if same candle confirms (closes back above)
                        if close > pdl:
                            risk = close - sweep_low_price
                            if risk > 0:
                                df.at[idx, "signal"] = 1
                                df.at[idx, "stop_loss"] = sweep_low_price
                                df.at[idx, "target"] = pdh
                                long_fired = True

                    elif low_swept:
                        # Update sweep extreme if price pushes lower
                        if low < sweep_low_price:
                            sweep_low_price = low

                        # Confirmation: closes back above prev day low
                        if close > pdl:
                            risk = close - sweep_low_price
                            if risk > 0:
                                df.at[idx, "signal"] = 1
                                df.at[idx, "stop_loss"] = sweep_low_price
                                df.at[idx, "target"] = pdh
                                long_fired = True

        return df
