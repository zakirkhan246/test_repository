"""
Backtesting engine for intraday trading strategies.

Simulates trade execution with:
- Candle-by-candle price checking against SL/target
- Intraday-only constraint (force close at 15:15)
- Maximum trades per day limit
- Realistic fill assumptions (entry at close of signal candle)
- Optional trailing stop modes: none, candle, atr, step
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

    Trailing modes:
      "none"   — fixed SL + fixed target (original behavior)
      "candle" — trail SL to prev candle's low (long) / high (short)
      "atr"    — trail SL at highest/lowest price minus N × ATR
      "step"   — move SL to breakeven at 1R, then trail in 1R steps
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 10.0,
        max_trades_per_day: int = 4,
        force_close_candle: int = 72,
        lot_size: int = 0,
        num_lots: int = 0,
        reversal_exit_pct: float = 0.0,
        trailing_mode: str = "none",
        atr_trail_multiplier: float = 2.0,
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_trades_per_day = max_trades_per_day
        self.force_close_candle = force_close_candle
        self.fixed_qty = lot_size * num_lots if (lot_size > 0 and num_lots > 0) else 0
        self.reversal_exit_pct = reversal_exit_pct
        self.trailing_mode = trailing_mode
        self.atr_trail_multiplier = atr_trail_multiplier

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on a DataFrame with signals.

        The DataFrame must have columns: open, high, low, close, signal,
        stop_loss, target, date, candle_idx
        For ATR trailing: also needs 'atr' column
        """
        trades = []
        open_trade = None
        daily_trade_count = {}
        capital = self.initial_capital

        dates = df["date"].unique()

        for date in dates:
            day_data = df[df["date"] == date]
            daily_trade_count[date] = 0
            prev_close = None
            prev_low = None
            prev_high = None

            for idx, (timestamp, row) in enumerate(day_data.iterrows()):
                if open_trade is not None:
                    # Update trailing SL before checking exit
                    if self.trailing_mode != "none":
                        self._update_trailing_sl(
                            open_trade, row, prev_close, prev_low, prev_high
                        )

                    # Check SL (and target if mode is "none")
                    trade_result = self._check_exit(open_trade, row, timestamp)

                    if trade_result is not None:
                        trade_result["duration_candles"] = (
                            row["candle_idx"] - open_trade["entry_candle_idx"]
                        )
                        trades.append(trade_result)
                        capital += trade_result["pnl"]
                        open_trade = None
                        prev_close = row["close"]
                        prev_low = row["low"]
                        prev_high = row["high"]
                        continue

                    # Reversal exit (only in fixed target mode)
                    if self.trailing_mode == "none" and self.reversal_exit_pct > 0:
                        trade_result = self._check_reversal_exit(
                            open_trade, row, prev_close, timestamp
                        )
                        if trade_result is not None:
                            trade_result["duration_candles"] = (
                                row["candle_idx"] - open_trade["entry_candle_idx"]
                            )
                            trades.append(trade_result)
                            capital += trade_result["pnl"]
                            open_trade = None
                            prev_close = row["close"]
                            prev_low = row["low"]
                            prev_high = row["high"]
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
                        prev_close = row["close"]
                        prev_low = row["low"]
                        prev_high = row["high"]
                        continue

                # Check for new entry signal
                if open_trade is None and row["signal"] != 0:
                    if daily_trade_count[date] < self.max_trades_per_day:
                        if not np.isnan(row["stop_loss"]) and not np.isnan(
                            row["target"]
                        ):
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
                                    "initial_sl": row["stop_loss"],
                                    "target": row["target"],
                                    "qty": qty,
                                    "entry_candle_idx": row["candle_idx"],
                                    "date": date,
                                    "mfe": 0.0,
                                    "risk": risk_per_unit,
                                    "best_price": row["close"],
                                }
                                daily_trade_count[date] += 1

                prev_close = row["close"]
                prev_low = row["low"]
                prev_high = row["high"]

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

        metrics = compute_metrics(trades, self.initial_capital)

        return {
            "trades": trades,
            "metrics": metrics,
            "final_capital": round(capital, 2),
        }

    def _update_trailing_sl(self, trade, candle, prev_close, prev_low, prev_high):
        """Update the trailing stop loss based on the selected mode."""
        direction = trade["direction"]
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        risk = trade["risk"]

        if direction == 1:
            trade["best_price"] = max(trade["best_price"], candle["high"])
        else:
            trade["best_price"] = min(trade["best_price"], candle["low"])

        if self.trailing_mode == "candle":
            if prev_low is None or prev_high is None:
                return
            if direction == 1:
                new_sl = prev_low
                if new_sl > sl:
                    trade["stop_loss"] = new_sl
            else:
                new_sl = prev_high
                if new_sl < sl:
                    trade["stop_loss"] = new_sl

        elif self.trailing_mode == "atr":
            atr_val = candle.get("atr", 0)
            if atr_val <= 0:
                return
            trail_dist = self.atr_trail_multiplier * atr_val
            if direction == 1:
                new_sl = trade["best_price"] - trail_dist
                if new_sl > sl:
                    trade["stop_loss"] = new_sl
            else:
                new_sl = trade["best_price"] + trail_dist
                if new_sl < sl:
                    trade["stop_loss"] = new_sl

        elif self.trailing_mode == "step":
            favorable = (trade["best_price"] - entry) if direction == 1 else (entry - trade["best_price"])
            r_multiple = favorable / risk if risk > 0 else 0
            if r_multiple >= 1.0:
                steps = int(r_multiple)
                if direction == 1:
                    new_sl = entry + (steps - 1) * risk
                    if new_sl > sl:
                        trade["stop_loss"] = new_sl
                else:
                    new_sl = entry - (steps - 1) * risk
                    if new_sl < sl:
                        trade["stop_loss"] = new_sl

    def _check_exit(
        self, trade: Dict, candle: pd.Series, timestamp
    ) -> Dict[str, Any] | None:
        """Check if current candle hits stop loss (and target in fixed mode)."""
        direction = trade["direction"]
        entry_price = trade["entry_price"]
        sl = trade["stop_loss"]
        target = trade["target"]
        qty = trade["qty"]

        if direction == 1:
            if candle["low"] <= sl:
                pnl = (sl - entry_price) * qty
                result = "WIN" if pnl > 0 else "LOSS"
                return self._build_trade_result(trade, sl, pnl, result, timestamp)

            if self.trailing_mode == "none" and candle["high"] >= target:
                pnl = (target - entry_price) * qty
                return self._build_trade_result(trade, target, pnl, "WIN", timestamp)

        elif direction == -1:
            if candle["high"] >= sl:
                pnl = (entry_price - sl) * qty
                result = "WIN" if pnl > 0 else "LOSS"
                return self._build_trade_result(trade, sl, pnl, result, timestamp)

            if self.trailing_mode == "none" and candle["low"] <= target:
                pnl = (entry_price - target) * qty
                return self._build_trade_result(trade, target, pnl, "WIN", timestamp)

        return None

    def _check_reversal_exit(
        self, trade: Dict, candle: pd.Series, prev_close, timestamp
    ) -> Dict[str, Any] | None:
        """Exit if trade reached reversal_exit_pct of target and candle closes against prev."""
        direction = trade["direction"]
        entry_price = trade["entry_price"]
        target = trade["target"]
        qty = trade["qty"]
        target_dist = abs(target - entry_price)
        if target_dist == 0:
            return None

        if direction == 1:
            trade["mfe"] = max(trade["mfe"], candle["high"] - entry_price)
        else:
            trade["mfe"] = max(trade["mfe"], entry_price - candle["low"])

        mfe_pct = trade["mfe"] / target_dist * 100
        if mfe_pct < self.reversal_exit_pct:
            return None

        if prev_close is None:
            return None

        is_reversal = False
        if direction == 1 and candle["close"] < prev_close:
            is_reversal = True
        elif direction == -1 and candle["close"] > prev_close:
            is_reversal = True

        if not is_reversal:
            return None

        exit_price = candle["close"]
        pnl = (exit_price - entry_price) * qty if direction == 1 else (entry_price - exit_price) * qty
        result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
        return self._build_trade_result(trade, exit_price, pnl, result, timestamp)

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
            "duration_candles": 0,
        }
