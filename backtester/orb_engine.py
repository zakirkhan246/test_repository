"""
ORB backtesting engine — wraps the SL/target BacktestEngine with trend filtering.

Takes the prepared ORB DataFrame (with signal, stop_loss, target, trend_score)
and zeroes out signals that don't pass the trend filter before running.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .engine import BacktestEngine


class ORBEngine:
    """
    Runs ORB backtest with optional trend filter.

    Delegates SL/target execution to BacktestEngine.
    """

    def __init__(
        self,
        initial_capital: float = 20000.0,
        max_trades_per_day: int = 4,
        lot_size: int = 10,
        num_lots: int = 2,
        use_trend_filter: bool = True,
        trend_threshold: int = 2,
    ):
        self.use_trend_filter = use_trend_filter
        self.trend_threshold = trend_threshold
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            max_trades_per_day=max_trades_per_day,
            lot_size=lot_size,
            num_lots=num_lots,
        )

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = df.copy()
        trend_blocked = 0

        if self.use_trend_filter:
            for idx in df.index:
                sig = df.at[idx, "signal"]
                if sig == 0:
                    continue
                score = df.at[idx, "trend_score"]
                if pd.isna(score):
                    df.at[idx, "signal"] = 0
                    df.at[idx, "stop_loss"] = np.nan
                    df.at[idx, "target"] = np.nan
                    trend_blocked += 1
                    continue
                allowed = (
                    (sig == 1 and score >= self.trend_threshold)
                    or (sig == -1 and score <= -self.trend_threshold)
                )
                if not allowed:
                    df.at[idx, "signal"] = 0
                    df.at[idx, "stop_loss"] = np.nan
                    df.at[idx, "target"] = np.nan
                    trend_blocked += 1

        result = self.engine.run(df)
        result["trend_blocked"] = trend_blocked
        return result
