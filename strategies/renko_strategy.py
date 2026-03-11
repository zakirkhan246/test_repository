"""
Renko Trend Pullback Continuation Strategy.

Chart:     Renko (fixed brick size, default 40 pts)
Market:    BSE Sensex Index — intraday only

Trend:     4+ consecutive same-direction bricks
Pullback:  Exactly 1 brick in opposite direction
Entry:     Close of 2nd continuation brick after pullback
Stop Loss: 1 opposite brick after entry
Target:    Trailing — exit on 2 consecutive opposite bricks
Re-entry:  Must wait for fresh 4-brick trend sequence
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import datetime


# ───────────────────────────────────────────────────────────
#  Renko Brick Builder
# ───────────────────────────────────────────────────────────

def build_renko_bricks(
    df: pd.DataFrame,
    brick_size: int = 40,
) -> pd.DataFrame:
    """
    Convert OHLC data into Renko bricks.

    Each brick is exactly `brick_size` points.  We walk through every
    close price and emit one or more bricks whenever the move from the
    current brick base exceeds one brick.

    Returns a DataFrame with columns:
        open, close, direction (+1=green, -1=red),
        timestamp (time the brick completed), date
    """
    closes = df["close"].values
    timestamps = df.index
    dates = df["date"].values

    if len(closes) == 0:
        return pd.DataFrame(columns=["open", "close", "direction", "timestamp", "date"])

    # Initialise from first close, rounded to nearest brick
    base = round(closes[0] / brick_size) * brick_size
    bricks: List[Dict] = []

    for i in range(1, len(closes)):
        price = closes[i]
        diff = price - base

        # How many full bricks can we make?
        num_bricks = int(abs(diff) // brick_size)
        if num_bricks == 0:
            continue

        direction = 1 if diff > 0 else -1

        for _ in range(num_bricks):
            brick_open = base
            base += direction * brick_size
            bricks.append({
                "open": brick_open,
                "close": base,
                "direction": direction,
                "timestamp": timestamps[i],
                "date": dates[i],
            })

    return pd.DataFrame(bricks)


# ───────────────────────────────────────────────────────────
#  Pattern Detector + Signal Generator
# ───────────────────────────────────────────────────────────

def detect_signals(bricks: pd.DataFrame, trend_len: int = 4) -> pd.DataFrame:
    """
    Scan Renko bricks for the Trend-Pullback-Continuation pattern.

    Pattern (long example):
        4+ green → 1 red → 2 green  →  LONG entry at close of 2nd green

    Returns bricks DataFrame with added 'signal' column:
        +1 = long entry, -1 = short entry, 0 = no signal
    """
    bricks = bricks.copy()
    bricks["signal"] = 0
    dirs = bricks["direction"].values
    n = len(dirs)

    i = 0
    while i < n:
        # ── Step 1: Find 4+ consecutive same-direction bricks ──
        trend_dir = dirs[i]
        trend_start = i
        while i < n and dirs[i] == trend_dir:
            i += 1
        trend_count = i - trend_start

        if trend_count < trend_len:
            continue  # not enough for a valid trend

        # ── Step 2: Exactly 1 pullback brick ──
        if i >= n or dirs[i] == trend_dir:
            continue  # no pullback
        pullback_idx = i
        i += 1
        # If more than 1 pullback brick → pattern broken
        if i < n and dirs[i] != trend_dir:
            continue

        # ── Step 3: 2 continuation bricks ──
        cont_count = 0
        while i < n and dirs[i] == trend_dir and cont_count < 2:
            cont_count += 1
            i += 1

        if cont_count == 2:
            # Entry at the close of the 2nd continuation brick
            entry_idx = i - 1
            bricks.iloc[entry_idx, bricks.columns.get_loc("signal")] = trend_dir

    return bricks


# ───────────────────────────────────────────────────────────
#  Renko Backtest Engine
# ───────────────────────────────────────────────────────────

class RenkoBacktestEngine:
    """
    Brick-by-brick backtest engine for the Renko Pullback strategy.

    Entry:   At close of signal brick
    SL:      1 opposite brick, but only if the trade is in a loss
             (current close is worse than entry). If in profit,
             the opposite brick is tolerated as a pullback.
    Target:  Trailing — exit when 2 consecutive opposite bricks appear.
    Exit:    Also force-close at EOD (intraday only).
    Re-entry: Must wait for new 4-brick trend after exit.
    """

    def __init__(
        self,
        brick_size: int = 40,
        qty: int = 20,
        initial_capital: float = 100_000.0,
    ):
        self.brick_size = brick_size
        self.qty = qty
        self.initial_capital = initial_capital

    def run(self, bricks: pd.DataFrame) -> Dict:
        """
        Run the Renko pullback backtest.

        `bricks` must have columns: open, close, direction, signal,
                                     timestamp, date
        """
        trades: List[Dict] = []
        dirs = bricks["direction"].values
        signals = bricks["signal"].values
        opens = bricks["open"].values
        closes = bricks["close"].values
        dates = bricks["date"].values
        timestamps = bricks["timestamp"].values
        n = len(bricks)

        open_trade = None
        opposite_count = 0       # consecutive opposite bricks
        waiting_for_new_trend = False

        for i in range(n):
            cur_dir = dirs[i]
            cur_open = opens[i]
            cur_close = closes[i]
            cur_date = dates[i]
            cur_ts = timestamps[i]

            # ── Force close at end of day ──
            if open_trade is not None and cur_date != open_trade["date"]:
                prev_close = closes[i - 1]
                prev_ts = timestamps[i - 1]
                pnl_pts = (prev_close - open_trade["entry_price"]) * open_trade["direction"]
                trades.append(self._close_trade(open_trade, prev_close, prev_ts, pnl_pts, "EOD_EXIT"))
                open_trade = None
                opposite_count = 0
                waiting_for_new_trend = True

            # ── Manage open trade ──
            if open_trade is not None:
                trade_dir = open_trade["direction"]
                entry_price = open_trade["entry_price"]

                # Current P&L at this brick's close
                cur_pnl = (cur_close - entry_price) * trade_dir
                in_loss = cur_pnl < 0

                # Track consecutive opposite bricks
                if cur_dir == trade_dir:
                    opposite_count = 0
                else:
                    opposite_count += 1

                # SL rule: 1 opposite brick, but ONLY if trade is in a loss.
                # This lets the brick fully form before deciding.
                # If in profit, we tolerate the pullback and let
                # the 2-brick trailing exit handle it.
                if opposite_count >= 1 and in_loss:
                    pnl_pts = (cur_close - entry_price) * trade_dir
                    trades.append(self._close_trade(
                        open_trade, cur_close, cur_ts, pnl_pts, "STOP_LOSS"))
                    open_trade = None
                    opposite_count = 0
                    waiting_for_new_trend = True
                    continue

                # Trailing exit: 2 consecutive opposite bricks (any P&L)
                if opposite_count >= 2:
                    pnl_pts = (cur_close - entry_price) * trade_dir
                    trades.append(self._close_trade(
                        open_trade, cur_close, cur_ts, pnl_pts, "TRAILING_EXIT"))
                    open_trade = None
                    opposite_count = 0
                    waiting_for_new_trend = True
                    continue

            # ── Check for new entry ──
            if open_trade is None and signals[i] != 0:
                if not waiting_for_new_trend:
                    trade_dir = int(signals[i])
                    open_trade = {
                        "entry_price": cur_close,
                        "entry_time": cur_ts,
                        "direction": trade_dir,
                        "date": cur_date,
                    }
                    opposite_count = 0

            # ── Reset re-entry lock when a fresh signal appears ──
            if waiting_for_new_trend and signals[i] != 0:
                if open_trade is None:
                    trade_dir = int(signals[i])
                    open_trade = {
                        "entry_price": cur_close,
                        "entry_time": cur_ts,
                        "direction": trade_dir,
                        "date": cur_date,
                    }
                    opposite_count = 0
                    waiting_for_new_trend = False

        # Close any remaining trade
        if open_trade is not None:
            pnl_pts = (closes[-1] - open_trade["entry_price"]) * open_trade["direction"]
            trades.append(self._close_trade(open_trade, closes[-1], timestamps[-1], pnl_pts, "EOD_EXIT"))

        return self._compute_results(trades)

    def _close_trade(self, trade, exit_price, exit_time, pnl_pts, exit_type):
        pnl_inr = pnl_pts * self.qty
        risk_pts = self.brick_size  # SL = 1 brick
        reward_risk = pnl_pts / risk_pts if risk_pts > 0 else 0
        return {
            "entry_time": trade["entry_time"],
            "exit_time": exit_time,
            "entry_price": trade["entry_price"],
            "exit_price": exit_price,
            "direction": trade["direction"],
            "pnl_pts": round(pnl_pts, 2),
            "pnl_inr": round(pnl_inr, 2),
            "reward_risk": round(reward_risk, 2),
            "result": "WIN" if pnl_pts > 0 else ("LOSS" if pnl_pts < 0 else "BREAKEVEN"),
            "exit_type": exit_type,
            "date": trade["date"],
        }

    def _compute_results(self, trades: List[Dict]) -> Dict:
        if not trades:
            return {"trades": [], "metrics": self._empty_metrics()}

        tdf = pd.DataFrame(trades)
        wins = tdf[tdf["result"] == "WIN"]
        losses = tdf[tdf["result"] != "WIN"]

        total = len(tdf)
        n_wins = len(wins)
        win_rate = n_wins / total * 100

        gross_profit = wins["pnl_inr"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_inr"].sum()) if len(losses) > 0 else 0
        net_pnl = tdf["pnl_inr"].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_rr = tdf["reward_risk"].mean()
        avg_profit_per_trade = net_pnl / total

        # Max drawdown
        equity = [self.initial_capital]
        for pnl in tdf["pnl_inr"]:
            equity.append(equity[-1] + pnl)
        eq = pd.Series(equity)
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        max_dd = abs(dd.min())

        metrics = {
            "total_trades": total,
            "wins": n_wins,
            "losses": total - n_wins,
            "win_rate": round(win_rate, 2),
            "avg_reward_to_risk": round(avg_rr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_profit_per_trade_inr": round(avg_profit_per_trade, 2),
            "profit_factor": round(profit_factor, 2),
            "net_pnl_pts": round(tdf["pnl_pts"].sum(), 2),
            "net_pnl_inr": round(net_pnl, 2),
            "gross_profit_inr": round(gross_profit, 2),
            "gross_loss_inr": round(gross_loss, 2),
            "long_trades": len(tdf[tdf["direction"] == 1]),
            "short_trades": len(tdf[tdf["direction"] == -1]),
            "long_win_rate": round(
                len(wins[wins["direction"] == 1]) / max(len(tdf[tdf["direction"] == 1]), 1) * 100, 2
            ),
            "short_win_rate": round(
                len(wins[wins["direction"] == -1]) / max(len(tdf[tdf["direction"] == -1]), 1) * 100, 2
            ),
        }

        return {"trades": trades, "metrics": metrics}

    def _empty_metrics(self):
        return {k: 0 for k in [
            "total_trades", "wins", "losses", "win_rate",
            "avg_reward_to_risk", "max_drawdown_pct",
            "avg_profit_per_trade_inr", "profit_factor",
            "net_pnl_pts", "net_pnl_inr",
            "gross_profit_inr", "gross_loss_inr",
            "long_trades", "short_trades",
            "long_win_rate", "short_win_rate",
        ]}
