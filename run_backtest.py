#!/usr/bin/env python3
"""
Backtest runner for pure MACD crossover strategy on real SENSEX data.

Capital: 20,000 INR | Lot size: 10 | Lots: 2 (qty = 20 per trade)
Timeframe: 5-minute candles | Risk-Reward: 1:2
"""

import sys
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_sensex_data
from strategies import MACDCrossover
from backtester import BacktestEngine


INITIAL_CAPITAL = 20_000.0
LOT_SIZE = 10
NUM_LOTS = 2


def weekly_pnl_table(trades, initial_capital):
    """Build a week-by-week P&L breakdown from trade list."""
    if not trades:
        return []

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["week_start"] = df["entry_time"].dt.to_period("W").apply(lambda p: p.start_time)

    rows = []
    running_capital = initial_capital

    for week, group in df.groupby("week_start"):
        wins = len(group[group["result"] == "WIN"])
        losses = len(group[group["result"].isin(["LOSS", "TIME_EXIT"])])
        week_pnl = group["pnl"].sum()
        running_capital += week_pnl
        n_days = group["date"].nunique()

        rows.append({
            "week": week.strftime("%d-%b-%Y"),
            "days": n_days,
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "win_rate": f"{wins / len(group) * 100:.0f}%" if len(group) > 0 else "0%",
            "pnl": round(week_pnl, 2),
            "capital": round(running_capital, 2),
        })

    return rows


def main():
    print("=" * 74)
    print("  SENSEX INTRADAY BACKTEST — Pure MACD Crossover")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lots × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty/trade")
    print("  Timeframe: 5-min candles | ATR stop | 1:2 RR")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    # Load real data
    data = load_sensex_data()
    n_days = data["date"].nunique()
    print(f"\n  Data: {n_days} trading days of real SENSEX 5-min candles")
    print(f"  Range: {data['date'].min()} to {data['date'].max()}")

    strategy = MACDCrossover(atr_multiplier=1.0, rr_ratio=2.0)
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=4,
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
    )

    # Run
    signals_df = strategy.run(data)
    result = engine.run(signals_df)
    trades = result["trades"]
    m = result["metrics"]

    # ── WEEKLY P&L ──
    weeks = weekly_pnl_table(trades, INITIAL_CAPITAL)

    print(f"\n{'─' * 74}")
    print("  WEEKLY P&L")
    print(f"{'─' * 74}\n")

    table = []
    for w in weeks:
        pnl_str = f"₹{w['pnl']:+,.2f}"
        cap_str = f"₹{w['capital']:,.2f}"
        table.append([
            w["week"], w["days"], w["trades"], w["wins"], w["losses"],
            w["win_rate"], pnl_str, cap_str,
        ])

    headers = ["Week Of", "Days", "Trades", "W", "L", "Win%", "P&L", "Capital"]
    print(tabulate(table, headers=headers, tablefmt="simple", stralign="right"))

    # Best / worst weeks
    if weeks:
        best_w = max(weeks, key=lambda w: w["pnl"])
        worst_w = min(weeks, key=lambda w: w["pnl"])
        green_weeks = sum(1 for w in weeks if w["pnl"] > 0)
        red_weeks = sum(1 for w in weeks if w["pnl"] <= 0)
        print(f"\n  Green weeks: {green_weeks} | Red weeks: {red_weeks}")
        print(f"  Best week:  {best_w['week']}  ₹{best_w['pnl']:+,.2f}")
        print(f"  Worst week: {worst_w['week']}  ₹{worst_w['pnl']:+,.2f}")

    # ── SUMMARY ──
    profitable = "YES" if m["win_rate"] >= 33.3 else "NO"

    print(f"\n{'─' * 74}")
    print("  OVERALL SUMMARY")
    print(f"{'─' * 74}")
    print(f"  Starting Capital:  ₹{INITIAL_CAPITAL:>10,.2f}")
    print(f"  Final Capital:     ₹{result['final_capital']:>10,.2f}")
    print(f"  Net P&L:           ₹{m['net_pnl']:>+10,.2f}  ({m['net_pnl_pct']:+.2f}%)")
    print(f"  Total Trades:      {m['total_trades']}")
    print(f"  Win Rate:          {m['win_rate']:.1f}%  (breakeven at 33.3% for 1:2 RR)")
    print(f"  Profitable:        {profitable}")
    print(f"  Profit Factor:     {m['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Estimate:   {m['sharpe_estimate']:.2f}")
    print(f"\n  Avg Win:           ₹{m['avg_win_pnl']:+,.2f}  ({m['avg_win_duration_candles']:.1f} candles / {m['avg_win_duration_candles'] * 5:.0f} min)")
    print(f"  Avg Loss:          ₹{m['avg_loss_pnl']:+,.2f}  ({m['avg_loss_duration_candles']:.1f} candles / {m['avg_loss_duration_candles'] * 5:.0f} min)")
    print(f"\n  Long Trades:       {m['long_trades']}  (Win: {m['long_win_rate']:.1f}%)")
    print(f"  Short Trades:      {m['short_trades']}  (Win: {m['short_win_rate']:.1f}%)")
    print(f"  Max Consec Wins:   {m['max_consecutive_wins']}")
    print(f"  Max Consec Losses: {m['max_consecutive_losses']}")

    print(f"\n{'─' * 74}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print("  Past performance does not guarantee future results.")
    print(f"{'─' * 74}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "strategy": "MACD Crossover",
            "config": {
                "initial_capital": INITIAL_CAPITAL,
                "lot_size": LOT_SIZE,
                "num_lots": NUM_LOTS,
                "qty_per_trade": LOT_SIZE * NUM_LOTS,
                "rr_ratio": "1:2",
                "atr_multiplier": 1.0,
                "max_trades_per_day": 4,
                "data_interval": "5min",
                "data_source": "real SENSEX 1-min resampled to 5-min",
                "trading_days": n_days,
            },
            "weekly_pnl": weeks,
            "metrics": m,
            "final_capital": result["final_capital"],
        }, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    return result


if __name__ == "__main__":
    main()
