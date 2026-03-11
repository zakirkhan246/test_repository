#!/usr/bin/env python3
"""
Backtest runner for Candle Range Theory (CRT) strategy on real SENSEX data.

Strategy: Detect liquidity sweeps of the previous day's high/low,
enter in the opposite direction after confirmed false breakout.

SL at the sweep extreme, target at the opposite end of prev day's range.

Tests both Regular and Heikin Ashi candles side-by-side.

Capital: 20,000 INR | Lot size: 10 | Lots: 2 (qty = 20 per trade)
Timeframe: 5-minute candles
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
from strategies.crt_strategy import CRTStrategy
from backtester.engine import BacktestEngine


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


def print_results(label, result, initial_capital):
    """Print full results for a backtest run."""
    trades = result["trades"]
    m = result["metrics"]

    weeks = weekly_pnl_table(trades, initial_capital)

    print(f"\n{'─' * 74}")
    print(f"  WEEKLY P&L — {label}")
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

    if weeks:
        best_w = max(weeks, key=lambda w: w["pnl"])
        worst_w = min(weeks, key=lambda w: w["pnl"])
        green_weeks = sum(1 for w in weeks if w["pnl"] > 0)
        red_weeks = sum(1 for w in weeks if w["pnl"] <= 0)
        print(f"\n  Green weeks: {green_weeks} | Red weeks: {red_weeks}")
        print(f"  Best week:  {best_w['week']}  ₹{best_w['pnl']:+,.2f}")
        print(f"  Worst week: {worst_w['week']}  ₹{worst_w['pnl']:+,.2f}")

    if trades:
        trade_df = pd.DataFrame(trades)
        target_exits = len(trade_df[trade_df["result"] == "WIN"])
        sl_exits = len(trade_df[trade_df["result"] == "LOSS"])
        time_exits = len(trade_df[trade_df["result"] == "TIME_EXIT"])
        print(f"\n{'─' * 74}")
        print(f"  EXIT BREAKDOWN — {label}")
        print(f"{'─' * 74}")
        print(f"  Target hits (WIN): {target_exits}  |  SL hits (LOSS): {sl_exits}  |  Time exits (EOD): {time_exits}")

    print(f"\n{'─' * 74}")
    print(f"  SUMMARY — {label}")
    print(f"{'─' * 74}")
    print(f"  Starting Capital:  ₹{initial_capital:>10,.2f}")
    print(f"  Final Capital:     ₹{result['final_capital']:>10,.2f}")
    print(f"  Net P&L:           ₹{m['net_pnl']:>+10,.2f}  ({m['net_pnl_pct']:+.2f}%)")
    print(f"  Total Trades:      {m['total_trades']}")
    print(f"  Win Rate:          {m['win_rate']:.1f}%")
    print(f"  Profit Factor:     {m['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Estimate:   {m['sharpe_estimate']:.2f}")
    print(f"\n  Avg Win:           ₹{m['avg_win_pnl']:+,.2f}  ({m['avg_win_duration_candles']:.1f} candles / {m['avg_win_duration_candles'] * 5:.0f} min)")
    print(f"  Avg Loss:          ₹{m['avg_loss_pnl']:+,.2f}  ({m['avg_loss_duration_candles']:.1f} candles / {m['avg_loss_duration_candles'] * 5:.0f} min)")
    print(f"\n  Long Trades:       {m['long_trades']}  (Win: {m['long_win_rate']:.1f}%)")
    print(f"  Short Trades:      {m['short_trades']}  (Win: {m['short_win_rate']:.1f}%)")
    print(f"  Max Consec Wins:   {m['max_consecutive_wins']}")
    print(f"  Max Consec Losses: {m['max_consecutive_losses']}")

    return weeks


def run_candle_type(candle_label, data):
    """Run CRT backtest for a given candle type."""
    strategy = CRTStrategy()
    prepared = strategy.prepare(data)

    # Count signals
    total_signals = (prepared["signal"] != 0).sum()
    long_signals = (prepared["signal"] == 1).sum()
    short_signals = (prepared["signal"] == -1).sum()
    print(f"\n  {candle_label}: {total_signals} CRT signals ({long_signals} long sweeps, {short_signals} short sweeps)")

    # Sweep stats
    days_with_signals = prepared[prepared["signal"] != 0]["date"].nunique()
    total_days = prepared["date"].nunique() - 1  # minus first day (no prev data)
    print(f"  Sweep days: {days_with_signals} / {total_days} trading days ({days_with_signals / total_days * 100:.1f}%)")

    # Run backtest (pure CRT, no filters)
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=2,  # Max 1 per direction
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
    )
    result = engine.run(prepared)
    run_label = f"{candle_label} — CRT"
    weeks = print_results(run_label, result, INITIAL_CAPITAL)

    return {"result": result, "weeks": weeks}


def print_comparison_table(reg_run, ha_run):
    """Print head-to-head comparison: Regular vs Heikin Ashi CRT."""
    all_results = [
        ("Regular", reg_run["result"]),
        ("Heikin Ashi", ha_run["result"]),
    ]
    ms = [(label, r["metrics"], r) for label, r in all_results]

    print(f"\n{'═' * 64}")
    print("  CRT: REGULAR vs HEIKIN ASHI")
    print(f"{'═' * 64}")
    print(f"  {'Metric':<22} {'Regular':>18} {'Heikin Ashi':>18}")
    print(f"  {'─' * 22} {'─' * 18} {'─' * 18}")

    def row(name, key, fmt="d"):
        vals = [m[key] for _, m, _ in ms]
        if fmt == "d":
            print(f"  {name:<22} {vals[0]:>18} {vals[1]:>18}")
        elif fmt == "pct":
            print(f"  {name:<22} {vals[0]:>17.1f}% {vals[1]:>17.1f}%")
        elif fmt == "f2":
            print(f"  {name:<22} {vals[0]:>18.2f} {vals[1]:>18.2f}")
        elif fmt == "inr":
            print(f"  {name:<22} {'₹{:+,.0f}'.format(vals[0]):>18} {'₹{:+,.0f}'.format(vals[1]):>18}")

    row("Total Trades", "total_trades", "d")

    wins = [int(m["total_trades"] * m["win_rate"] / 100) for _, m, _ in ms]
    print(f"  {'Wins':<22} {wins[0]:>18} {wins[1]:>18}")

    row("Win Rate", "win_rate", "pct")
    row("Profit Factor", "profit_factor", "f2")
    row("Net P&L", "net_pnl", "inr")

    caps = [r["final_capital"] for _, _, r in ms]
    print(f"  {'Final Capital':<22} {'₹{:,.0f}'.format(caps[0]):>18} {'₹{:,.0f}'.format(caps[1]):>18}")

    row("Max Drawdown", "max_drawdown_pct", "pct")
    row("Sharpe Estimate", "sharpe_estimate", "f2")
    row("Avg Win", "avg_win_pnl", "inr")
    row("Avg Loss", "avg_loss_pnl", "inr")

    rrs = [abs(m["avg_win_pnl"] / m["avg_loss_pnl"]) if m["avg_loss_pnl"] != 0 else 0 for _, m, _ in ms]
    print(f"  {'Avg R:R':<22} {rrs[0]:>17.2f}x {rrs[1]:>17.2f}x")

    row("Max Consec Wins", "max_consecutive_wins", "d")
    row("Max Consec Losses", "max_consecutive_losses", "d")
    row("Long Trades", "long_trades", "d")
    row("Short Trades", "short_trades", "d")
    row("Long Win Rate", "long_win_rate", "pct")
    row("Short Win Rate", "short_win_rate", "pct")


def main():
    print("=" * 74)
    print("  SENSEX INTRADAY BACKTEST — Candle Range Theory (CRT)")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lots × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty/trade")
    print("  Range: Previous day's high/low")
    print("  Entry: Reversal after confirmed liquidity sweep")
    print("  SL: Sweep extreme | Target: Opposite end of prev day range")
    print("  No trend filter — pure CRT concept")
    print("  Candle types: Regular OHLC vs Heikin Ashi")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    # Load both candle types
    data_regular = load_sensex_data()
    data_ha = load_sensex_data(heikin_ashi=True)
    n_days = data_regular["date"].nunique()
    print(f"\n  Data: {n_days} trading days of real SENSEX 5-min candles")
    print(f"  Range: {data_regular['date'].min()} to {data_regular['date'].max()}")

    # ═══════════════════════════════════════════════════════════
    # Run Regular candles
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 74}")
    print("  PART 1: REGULAR CANDLES — CRT")
    print(f"{'═' * 74}")
    reg_run = run_candle_type("REGULAR", data_regular)

    # ═══════════════════════════════════════════════════════════
    # Run Heikin Ashi candles
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 74}")
    print("  PART 2: HEIKIN ASHI CANDLES — CRT")
    print(f"{'═' * 74}")
    ha_run = run_candle_type("HEIKIN ASHI", data_ha)

    # ═══════════════════════════════════════════════════════════
    # Head-to-head
    # ═══════════════════════════════════════════════════════════
    print_comparison_table(reg_run, ha_run)

    print(f"\n{'─' * 74}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print("  Past performance does not guarantee future results.")
    print(f"{'─' * 74}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "crt_backtest_results.json")
    save_data = {
        "run_date": datetime.now().isoformat(),
        "strategy": "Candle Range Theory (CRT)",
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "lot_size": LOT_SIZE,
            "num_lots": NUM_LOTS,
            "qty_per_trade": LOT_SIZE * NUM_LOTS,
            "range": "previous day high/low",
            "entry": "reversal after confirmed liquidity sweep",
            "sl": "sweep extreme (max high or min low during sweep)",
            "target": "opposite end of previous day range",
            "trend_filter": "none",
            "max_trades_per_day": 2,
            "data_interval": "5min",
            "data_source": "real SENSEX 1-min resampled to 5-min",
            "trading_days": n_days,
        },
    }
    for candle_type, run_data in [("regular", reg_run), ("heikin_ashi", ha_run)]:
        r = run_data["result"]
        save_data[candle_type] = {
            "weekly_pnl": run_data["weeks"],
            "metrics": r["metrics"],
            "final_capital": r["final_capital"],
        }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    return reg_run["result"]


if __name__ == "__main__":
    main()
