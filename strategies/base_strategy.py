"""
Base strategy class defining the interface for all trading strategies.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .indicators import compute_macd, compute_atr


class Signal:
    BUY = 1
    SELL = -1
    NONE = 0


class BaseStrategy(ABC):
    """
    Base class for intraday strategies.

    Enforces:
    - 1:2 risk-reward ratio (stop loss : target)
    - ATR-based stop loss calculation
    - No overnight positions
    """

    def __init__(
        self,
        atr_multiplier: float = 1.5,
        rr_ratio: float = 2.0,
        atr_period: int = 14,
        entry_start_candle: int = 6,   # Skip first 30 min (6 x 5min)
        entry_end_candle: int = 65,    # No new entries after 14:40
    ):
        self.atr_multiplier = atr_multiplier
        self.rr_ratio = rr_ratio
        self.atr_period = atr_period
        self.entry_start_candle = entry_start_candle
        self.entry_end_candle = entry_end_candle

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # MACD
        macd = compute_macd(df["close"])
        df["macd_line"] = macd["macd_line"]
        df["signal_line"] = macd["signal_line"]
        df["macd_hist"] = macd["histogram"]

        # ATR for stop-loss calculation
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.atr_period)

        # MACD crossover detection
        df["macd_cross_up"] = (df["macd_line"] > df["signal_line"]) & (
            df["macd_line"].shift(1) <= df["signal_line"].shift(1)
        )
        df["macd_cross_down"] = (df["macd_line"] < df["signal_line"]) & (
            df["macd_line"].shift(1) >= df["signal_line"].shift(1)
        )

        # MACD histogram direction
        df["macd_hist_rising"] = df["macd_hist"] > df["macd_hist"].shift(1)

        # Candle index within day
        df["candle_idx"] = df.groupby("date").cumcount()

        return df

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def apply_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["stop_loss"] = np.nan
        df["target"] = np.nan

        buy_mask = df["signal"] == Signal.BUY
        sell_mask = df["signal"] == Signal.SELL

        if buy_mask.any():
            risk = df.loc[buy_mask, "atr"] * self.atr_multiplier
            df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "close"] - risk
            df.loc[buy_mask, "target"] = (
                df.loc[buy_mask, "close"] + risk * self.rr_ratio
            )

        if sell_mask.any():
            risk = df.loc[sell_mask, "atr"] * self.atr_multiplier
            df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "close"] + risk
            df.loc[sell_mask, "target"] = (
                df.loc[sell_mask, "close"] - risk * self.rr_ratio
            )

        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_indicators(df)
        df = self.generate_signals(df)
        df = self.apply_risk_management(df)
        return df
