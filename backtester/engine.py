"""
Backtesting engine for intraday trading strategies.

Simulates trade execution with:
- Candle-by-candle price checking against SL/target
- Intraday-only constraint (force close at 15:15)
- Maximum trades per day limit
- Realistic fill assumptions (entry at close of signal candle)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .metrics import compute_metrics


class BacktestEngine:
    """
    Event-driven backtesting engine for intraday strategies.

    Processes each candle sequentially, manages open positions,
    and tracks all trades with full details.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 10.0,  # % of capital per trade (ignored if fixed_qty set)
        max_trades_per_day: int = 4,
        force_close_candle: int = 72,  # 15:15 (candle 72 of 75)
        lot_size: int = 0,  # Units per lot (e.g. 10 for SENSEX). 0 = use pct sizing
        num_lots: int = 0,  # Number of lots to trade. 0 = use pct sizing
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_trades_per_day = max_trades_per_day
        self.force_close_candle = force_close_candle
        self.fixed_qty = lot_size * num_lots if (lot_size > 0 and num_lots > 0) else 0

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on a DataFrame with signals.

        The DataFrame must have columns: open, high, low, close, signal,
        stop_loss, target, date, candle_idx
        """
        trades = []
        open_trade = None
        daily_trade_count = {}
        capital = self.initial_capital

        dates = df["date"].unique()

        for date in dates:
            day_data = df[df["date"] == date]
            daily_trade_count[date] = 0

            for idx, (timestamp, row) in enumerate(day_data.iterrows()):
                # Check if we have an open trade
                if open_trade is not None:
                    # Check stop loss and target against current candle
                    trade_result = self._check_exit(open_trade, row, timestamp)

                    if trade_result is not None:
                        trade_result["duration_candles"] = (
                            row["candle_idx"] - open_trade["entry_candle_idx"]
                        )
                        trades.append(trade_result)
                        capital += trade_result["pnl"]
                        open_trade = None
                        continue

                    # Force close at end of day
                    if row["candle_idx"] >= self.force_close_candle:
                        trade_result = self._force_close(open_trade, row, timestamp)
                        trade_result["duration_candles"] = (
                            row["candle_idx"] - open_trade["entry_candle_idx"]
                        )
                        trades.append(trade_result)
                        capital += trade_result["pnl"]
                        open_trade = None
                        continue

                # Check for new entry signal (only if no open trade)
                if open_trade is None and row["signal"] != 0:
                    if daily_trade_count[date] < self.max_trades_per_day:
                        if not np.isnan(row["stop_loss"]) and not np.isnan(
                            row["target"]
                        ):
                            # Calculate position size
                            risk_per_unit = abs(row["close"] - row["stop_loss"])
                            if risk_per_unit > 0:
                                if self.fixed_qty > 0:
                                    qty = self.fixed_qty
                                else:
                                    trade_capital = capital * (
                                        self.position_size_pct / 100
                                    )
                                    qty = trade_capital / row["close"]

                                open_trade = {
                                    "entry_time": timestamp,
                                    "entry_price": row["close"],
                                    "direction": int(row["signal"]),
                                    "stop_loss": row["stop_loss"],
                                    "target": row["target"],
                                    "qty": qty,
                                    "entry_candle_idx": row["candle_idx"],
                                    "date": date,
                                }
                                daily_trade_count[date] += 1

            # End of day: force close any open trade
            if open_trade is not None:
                last_row = day_data.iloc[-1]
                last_ts = day_data.index[-1]
                trade_result = self._force_close(open_trade, last_row, last_ts)
                trade_result["duration_candles"] = (
                    last_row["candle_idx"] - open_trade["entry_candle_idx"]
                )
                trades.append(trade_result)
                capital += trade_result["pnl"]
                open_trade = None

        # Compute metrics
        metrics = compute_metrics(trades, self.initial_capital)

        return {
            "trades": trades,
            "metrics": metrics,
            "final_capital": round(capital, 2),
        }

    def _check_exit(
        self, trade: Dict, candle: pd.Series, timestamp
    ) -> Dict[str, Any] | None:
        """Check if current candle hits stop loss or target."""
        direction = trade["direction"]
        entry_price = trade["entry_price"]
        sl = trade["stop_loss"]
        target = trade["target"]
        qty = trade["qty"]

        if direction == 1:  # Long trade
            # Check stop loss first (conservative: assume worst case)
            if candle["low"] <= sl:
                pnl = (sl - entry_price) * qty
                return self._build_trade_result(trade, sl, pnl, "LOSS", timestamp)

            # Check target
            if candle["high"] >= target:
                pnl = (target - entry_price) * qty
                return self._build_trade_result(trade, target, pnl, "WIN", timestamp)

        elif direction == -1:  # Short trade
            # Check stop loss first
            if candle["high"] >= sl:
                pnl = (entry_price - sl) * qty
                return self._build_trade_result(trade, sl, pnl, "LOSS", timestamp)

            # Check target
            if candle["low"] <= target:
                pnl = (entry_price - target) * qty
                return self._build_trade_result(trade, target, pnl, "WIN", timestamp)

        return None

    def _force_close(
        self, trade: Dict, candle: pd.Series, timestamp
    ) -> Dict[str, Any]:
        """Force close at end of day or at force-close time."""
        direction = trade["direction"]
        entry_price = trade["entry_price"]
        exit_price = candle["close"]
        qty = trade["qty"]

        if direction == 1:
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
        return self._build_trade_result(
            trade, exit_price, pnl, "TIME_EXIT", timestamp
        )

    def _build_trade_result(
        self,
        trade: Dict,
        exit_price: float,
        pnl: float,
        result: str,
        exit_time,
    ) -> Dict[str, Any]:
        """Construct a trade result dictionary."""
        return {
            "entry_time": trade["entry_time"],
            "exit_time": exit_time,
            "entry_price": round(trade["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "direction": trade["direction"],
            "pnl": round(pnl, 2),
            "result": result,
            "date": trade["date"],
            "qty": trade["qty"],
            "stop_loss": trade["stop_loss"],
            "target": trade["target"],
            "duration_candles": 0,  # Will be overwritten by caller
        }
