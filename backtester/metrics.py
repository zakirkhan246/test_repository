"""
Performance metrics calculation for backtested trades.

Computes win rate, max drawdown, profit factor, average trade duration,
and other key metrics for strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def compute_metrics(trades: List[Dict[str, Any]], initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    Compute comprehensive performance metrics from a list of trades.

    Each trade dict must have:
    - entry_price, exit_price, direction (1=long, -1=short)
    - entry_time, exit_time
    - pnl, result ('WIN', 'LOSS', 'BREAKEVEN', 'TIME_EXIT')
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win_duration_candles": 0,
            "avg_loss_duration_candles": 0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "net_pnl_pct": 0.0,
            "avg_win_pnl": 0.0,
            "avg_loss_pnl": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "sharpe_estimate": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
        }

    df = pd.DataFrame(trades)

    # Basic counts
    total = len(df)
    wins = df[df["result"] == "WIN"]
    losses = df[df["result"].isin(["LOSS", "TIME_EXIT"])]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100 if total > 0 else 0

    # PnL metrics
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    net_pnl = df["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win_pnl = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss_pnl = losses["pnl"].mean() if len(losses) > 0 else 0

    # Trade duration
    avg_win_dur = wins["duration_candles"].mean() if len(wins) > 0 else 0
    avg_loss_dur = losses["duration_candles"].mean() if len(losses) > 0 else 0

    # Max drawdown from equity curve
    equity = [initial_capital]
    for pnl in df["pnl"]:
        equity.append(equity[-1] + pnl)
    equity = pd.Series(equity)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = abs(drawdown.min())

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for r in df["result"]:
        if r == "WIN":
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Sharpe estimate (daily PnL based)
    pnl_series = df["pnl"]
    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)) if pnl_series.std() > 0 else 0

    # Long/Short breakdown
    longs = df[df["direction"] == 1]
    shorts = df[df["direction"] == -1]
    long_wins = longs[longs["result"] == "WIN"]
    short_wins = shorts[shorts["result"] == "WIN"]

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 2),
        "loss_rate": round(100 - win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_win_duration_candles": round(avg_win_dur, 1),
        "avg_loss_duration_candles": round(avg_loss_dur, 1),
        "profit_factor": round(profit_factor, 2),
        "net_pnl": round(net_pnl, 2),
        "net_pnl_pct": round(net_pnl / initial_capital * 100, 2),
        "avg_win_pnl": round(avg_win_pnl, 2),
        "avg_loss_pnl": round(avg_loss_pnl, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "sharpe_estimate": round(sharpe, 2),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_win_rate": round(len(long_wins) / len(longs) * 100, 2) if len(longs) > 0 else 0,
        "short_win_rate": round(len(short_wins) / len(shorts) * 100, 2) if len(shorts) > 0 else 0,
    }
