"""
Channel-based backtesting engine for EMA Channel strategy.

Unlike the SL/target engine, this manages entries and exits based on
price position relative to the EMA channel:

- Tracks which "side" of the channel price is on (above/below)
- Long entry when price crosses from below to above
- Exit when price crosses back
- Re-entry in opposite direction if confirmed by next candle
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .metrics import compute_metrics


class ChannelEngine:
    """
    Event-driven backtesting engine for channel-based strategies.

    Processes each candle, tracks channel position, and manages
    entries/exits/re-entries based on close vs channel boundaries.
    """

    def __init__(
        self,
        initial_capital: float = 20000.0,
        max_trades_per_day: int = 4,
        lot_size: int = 10,
        num_lots: int = 2,
        entry_start_candle: int = 6,   # Skip first 30 min (6 × 5min)
        entry_end_candle: int = 65,    # No new entries after 14:40
        force_close_candle: int = 72,  # Force close at 15:15
    ):
        self.initial_capital = initial_capital
        self.max_trades_per_day = max_trades_per_day
        self.fixed_qty = lot_size * num_lots
        self.entry_start_candle = entry_start_candle
        self.entry_end_candle = entry_end_candle
        self.force_close_candle = force_close_candle

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run channel-based backtest.

        DataFrame must have columns:
        open, high, low, close, channel_upper, channel_lower, date, candle_idx
        """
        trades = []
        capital = self.initial_capital
        daily_trade_count = {}
        dates = df["date"].unique()

        for date in dates:
            day_data = df[df["date"] == date]
            daily_trade_count[date] = 0

            position = 0        # 0=flat, 1=long, -1=short
            open_trade = None
            just_exited = 0     # 1=exited long, -1=exited short
            exit_idx = -1       # Day-relative index of exit candle
            last_side = None    # "below" or "above" — last clear side of channel

            rows = list(day_data.iterrows())

            for i, (timestamp, row) in enumerate(rows):
                ch_upper = row["channel_upper"]
                ch_lower = row["channel_lower"]
                close = row["close"]
                candle_idx = row["candle_idx"]

                # ── Force close at end of day ──
                if open_trade is not None and candle_idx >= self.force_close_candle:
                    result = self._close_trade(
                        open_trade, row, timestamp, force_exit=True
                    )
                    trades.append(result)
                    capital += result["pnl"]
                    open_trade = None
                    position = 0
                    just_exited = 0
                    if close > ch_upper:
                        last_side = "above"
                    elif close < ch_lower:
                        last_side = "below"
                    continue

                # ── Check exits for open positions ──
                if position == 1 and close < ch_lower:
                    # Exit long: price closed below channel
                    result = self._close_trade(open_trade, row, timestamp)
                    trades.append(result)
                    capital += result["pnl"]
                    open_trade = None
                    position = 0
                    just_exited = 1
                    exit_idx = i
                    last_side = "below"
                    continue

                if position == -1 and close > ch_upper:
                    # Exit short: price closed above channel
                    result = self._close_trade(open_trade, row, timestamp)
                    trades.append(result)
                    capital += result["pnl"]
                    open_trade = None
                    position = 0
                    just_exited = -1
                    exit_idx = i
                    last_side = "above"
                    continue

                # ── Check entries (only if flat and in trading window) ──
                in_window = (
                    candle_idx >= self.entry_start_candle
                    and candle_idx <= self.entry_end_candle
                )
                can_trade = (
                    position == 0
                    and in_window
                    and daily_trade_count[date] < self.max_trades_per_day
                )

                if can_trade:
                    entered = False

                    # Re-entry: only on the candle immediately after exit
                    if just_exited != 0 and i == exit_idx + 1:
                        if just_exited == 1 and close < ch_lower:
                            # After long exit, next candle confirms below → SHORT
                            open_trade = self._open_trade(
                                row, timestamp, -1, date
                            )
                            position = -1
                            daily_trade_count[date] += 1
                            entered = True

                        elif just_exited == -1 and close > ch_upper:
                            # After short exit, next candle confirms above → LONG
                            open_trade = self._open_trade(
                                row, timestamp, 1, date
                            )
                            position = 1
                            daily_trade_count[date] += 1
                            entered = True

                    # Normal entry: channel crossover
                    elif just_exited == 0 or i > exit_idx + 1:
                        if last_side == "below" and close > ch_upper:
                            # Price crossed from below to above → LONG
                            open_trade = self._open_trade(
                                row, timestamp, 1, date
                            )
                            position = 1
                            daily_trade_count[date] += 1
                            entered = True

                        elif last_side == "above" and close < ch_lower:
                            # Price crossed from above to below → SHORT
                            open_trade = self._open_trade(
                                row, timestamp, -1, date
                            )
                            position = -1
                            daily_trade_count[date] += 1
                            entered = True

                    if entered:
                        just_exited = 0

                # Reset re-entry window after one candle
                if just_exited != 0 and i > exit_idx + 1:
                    just_exited = 0

                # Track which side of channel we're on
                if close > ch_upper:
                    last_side = "above"
                elif close < ch_lower:
                    last_side = "below"
                # Inside channel: keep previous last_side

            # End of day: force close any remaining position
            if open_trade is not None:
                last_row = day_data.iloc[-1]
                last_ts = day_data.index[-1]
                result = self._close_trade(
                    open_trade, last_row, last_ts, force_exit=True
                )
                trades.append(result)
                capital += result["pnl"]
                open_trade = None

        metrics = compute_metrics(trades, self.initial_capital)
        return {
            "trades": trades,
            "metrics": metrics,
            "final_capital": round(capital, 2),
        }

    def _open_trade(self, row, timestamp, direction, date) -> Dict[str, Any]:
        """Create a new trade entry."""
        return {
            "entry_time": timestamp,
            "entry_price": row["close"],
            "direction": direction,
            "qty": self.fixed_qty,
            "entry_candle_idx": row["candle_idx"],
            "date": date,
        }

    def _close_trade(
        self, trade: Dict, row, timestamp, force_exit: bool = False
    ) -> Dict[str, Any]:
        """Close an open trade and compute P&L."""
        direction = trade["direction"]
        entry_price = trade["entry_price"]
        exit_price = row["close"]
        qty = trade["qty"]

        if direction == 1:
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        if force_exit:
            result = "TIME_EXIT"
        else:
            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

        return {
            "entry_time": trade["entry_time"],
            "exit_time": timestamp,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "direction": direction,
            "pnl": round(pnl, 2),
            "result": result,
            "date": trade["date"],
            "qty": trade["qty"],
            "stop_loss": None,
            "target": None,
            "duration_candles": row["candle_idx"] - trade["entry_candle_idx"],
        }

