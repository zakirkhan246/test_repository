"""
Candle Range Theory (CRT) Strategy — Hourly Timeframe.

CRT is a Smart Money / ICT reversal strategy based on false breakouts
(liquidity sweeps) of the previous hour's range.

Range: Previous 1-hour candle's high and low (the "dealing range").
       Resets every 12 five-minute candles throughout the trading day.

Sweep Detection:
  HIGH SWEEP: Candle wick pierces above prev_hour_high (liquidity grab)
  LOW SWEEP:  Candle wick pierces below prev_hour_low (liquidity grab)

Confirmation & Entry:
  After high sweep → candle closes back BELOW prev_hour_high → SHORT
  After low sweep  → candle closes back ABOVE prev_hour_low  → LONG

Risk Management:
  SHORT: SL = sweep high (max high during sweep), Target = prev_hour_low
  LONG:  SL = sweep low (min low during sweep),  Target = prev_hour_high

Max 1 trade per direction per hourly block.
"""

import pandas as pd
import numpy as np
from .indicators import compute_atr


class CRTStrategy:
    """
    Computes CRT (Candle Range Theory) signals using the previous
    1-hour candle's range and detecting liquidity sweeps on 5-min candles.
    """

    def __init__(
        self,
        candles_per_hour: int = 12,    # 12 × 5min = 1 hour
        atr_period: int = 14,
        entry_end_candle: int = 65,    # No new entries after 14:40
    ):
        self.candles_per_hour = candles_per_hour
        self.atr_period = atr_period
        self.entry_end_candle = entry_end_candle

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute previous hour's range, detect sweeps, generate signals."""
        df = df.copy()

        # Candle index within day
        df["candle_idx"] = df.groupby("date").cumcount()

        # ── Assign hourly block index within each day ──
        # Block 0: candles 0-11 (9:15-10:15)
        # Block 1: candles 12-23 (10:15-11:15)
        # Block 2: candles 24-35 (11:15-12:15)
        # etc.
        df["hour_block"] = df["candle_idx"] // self.candles_per_hour

        # ATR for reference
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.atr_period)

        # ── Compute previous hour's high/low ──
        # Group by (date, hour_block) to get each hour's range
        hourly_hl = df.groupby(["date", "hour_block"]).agg(
            hour_high=("high", "max"),
            hour_low=("low", "min"),
        ).reset_index()

        # Map previous hour's high/low to each candle
        df["prev_hour_high"] = np.nan
        df["prev_hour_low"] = np.nan

        for date in df["date"].unique():
            day_hours = hourly_hl[hourly_hl["date"] == date].sort_values("hour_block")
            day_mask = df["date"] == date

            for i, hour_row in day_hours.iterrows():
                block = hour_row["hour_block"]
                if block == 0:
                    # First hour has no previous hour — skip
                    continue

                # Get previous hour's range
                prev = day_hours[day_hours["hour_block"] == block - 1]
                if prev.empty:
                    continue

                prev_high = prev["hour_high"].iloc[0]
                prev_low = prev["hour_low"].iloc[0]

                # Apply to all candles in this block
                block_mask = day_mask & (df["hour_block"] == block)
                df.loc[block_mask, "prev_hour_high"] = prev_high
                df.loc[block_mask, "prev_hour_low"] = prev_low

        df["prev_hour_range"] = df["prev_hour_high"] - df["prev_hour_low"]

        # ── Sweep Detection & Signal Generation ──
        df["signal"] = 0
        df["stop_loss"] = np.nan
        df["target"] = np.nan

        for date in df["date"].unique():
            day_df = df[df["date"] == date]
            hour_blocks = sorted(day_df["hour_block"].unique())

            for block in hour_blocks:
                block_df = day_df[day_df["hour_block"] == block]

                # Skip if no previous hour data
                if pd.isna(block_df["prev_hour_high"].iloc[0]):
                    continue

                phh = block_df["prev_hour_high"].iloc[0]
                phl = block_df["prev_hour_low"].iloc[0]

                # State tracking — resets each hour
                high_swept = False
                low_swept = False
                short_fired = False
                long_fired = False
                sweep_high_price = np.nan
                sweep_low_price = np.nan

                for idx in block_df.index:
                    row = df.loc[idx]
                    ci = row["candle_idx"]

                    # No entries too late in the day
                    if ci > self.entry_end_candle:
                        continue

                    high = row["high"]
                    low = row["low"]
                    close = row["close"]

                    # ── HIGH SWEEP (bearish setup) ──
                    if not short_fired:
                        if not high_swept and high > phh:
                            high_swept = True
                            sweep_high_price = high

                            if close < phh:
                                risk = sweep_high_price - close
                                if risk > 0:
                                    df.at[idx, "signal"] = -1
                                    df.at[idx, "stop_loss"] = sweep_high_price
                                    df.at[idx, "target"] = phl
                                    short_fired = True

                        elif high_swept:
                            if high > sweep_high_price:
                                sweep_high_price = high

                            if close < phh:
                                risk = sweep_high_price - close
                                if risk > 0:
                                    df.at[idx, "signal"] = -1
                                    df.at[idx, "stop_loss"] = sweep_high_price
                                    df.at[idx, "target"] = phl
                                    short_fired = True

                    # ── LOW SWEEP (bullish setup) ──
                    if not long_fired:
                        if not low_swept and low < phl:
                            low_swept = True
                            sweep_low_price = low

                            if close > phl:
                                risk = close - sweep_low_price
                                if risk > 0:
                                    df.at[idx, "signal"] = 1
                                    df.at[idx, "stop_loss"] = sweep_low_price
                                    df.at[idx, "target"] = phh
                                    long_fired = True

                        elif low_swept:
                            if low < sweep_low_price:
                                sweep_low_price = low

                            if close > phl:
                                risk = close - sweep_low_price
                                if risk > 0:
                                    df.at[idx, "signal"] = 1
                                    df.at[idx, "stop_loss"] = sweep_low_price
                                    df.at[idx, "target"] = phh
                                    long_fired = True

        return df
