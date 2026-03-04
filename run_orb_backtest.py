#!/usr/bin/env python3
"""
Backtest runner for Opening Range Breakout (ORB) strategy on real SENSEX data.

Strategy: Enter when price breaks above/below the 30-min opening range.
SL at the opposite OR boundary, target at 1:2 R:R.

Trend Filter: 3-factor composite score (VWAP + OR position + EMA Slope).
Only trade in the direction of the day's trend (score >= 2 or <= -2).

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
from strategies.orb_strategy import ORBStrategy
from backtester.orb_engine import ORBEngine


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
        sl_exits = len(trade_df[trade_df["result"] == "LOSS"])
        target_exits = len(trade_df[trade_df["result"] == "WIN"])
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

    if result.get("trend_blocked", 0) > 0:
        print(f"\n  Trend-filtered:    {result['trend_blocked']} entries blocked by trend filter")

    return weeks


def run_candle_type(candle_label, data):
    """Run ORB backtest for a given candle type (Regular or Heikin Ashi)."""
    strategy = ORBStrategy(or_candles=6, rr_ratio=2.0)
    prepared = strategy.prepare(data)

    # Count signals before filtering
    total_signals = (prepared["signal"] != 0).sum()
    long_signals = (prepared["signal"] == 1).sum()
    short_signals = (prepared["signal"] == -1).sum()
    print(f"\n  {candle_label}: {total_signals} raw signals ({long_signals} long, {short_signals} short)")

    # Trend score distribution on signal candles
    signal_candles = prepared[prepared["signal"] != 0]
    if len(signal_candles) > 0:
        score_dist = signal_candles["trend_score"].value_counts().sort_index()
        print(f"  Trend scores on signal candles:")
        for score, count in score_dist.items():
            print(f"    Score {score:+d}: {count:>4}")

    # Run with trend filter and without
    runs = {}
    for label, use_filter in [
        ("WITH TREND FILTER", True),
        ("NO FILTER (baseline)", False),
    ]:
        engine = ORBEngine(
            initial_capital=INITIAL_CAPITAL,
            max_trades_per_day=4,
            lot_size=LOT_SIZE,
            num_lots=NUM_LOTS,
            use_trend_filter=use_filter,
            trend_threshold=2,
        )
        result = engine.run(prepared)
        run_label = f"{candle_label} ORB — {label}"
        weeks = print_results(run_label, result, INITIAL_CAPITAL)
        runs[label] = {"result": result, "weeks": weeks}

    return runs


def print_comparison_table(reg_runs, ha_runs):
    """Print head-to-head comparison: Regular vs Heikin Ashi ORB."""
    all_results = [
        ("Reg+Filter", reg_runs["WITH TREND FILTER"]["result"]),
        ("Reg NoFilter", reg_runs["NO FILTER (baseline)"]["result"]),
        ("HA+Filter", ha_runs["WITH TREND FILTER"]["result"]),
        ("HA NoFilter", ha_runs["NO FILTER (baseline)"]["result"]),
    ]
    ms = [(label, r["metrics"], r) for label, r in all_results]

    print(f"\n{'═' * 98}")
    print("  ORB: REGULAR vs HEIKIN ASHI — HEAD-TO-HEAD COMPARISON")
    print(f"{'═' * 98}")
    print(f"  {'Metric':<22} {'Reg+Filter':>16} {'Reg NoFilter':>16} {'HA+Filter':>16} {'HA NoFilter':>16}")
    print(f"  {'─' * 22} {'─' * 16} {'─' * 16} {'─' * 16} {'─' * 16}")

    def row(name, key, fmt="d"):
        vals = [m[key] for _, m, _ in ms]
        if fmt == "d":
            print(f"  {name:<22} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16} {vals[3]:>16}")
        elif fmt == "pct":
            print(f"  {name:<22} {vals[0]:>15.1f}% {vals[1]:>15.1f}% {vals[2]:>15.1f}% {vals[3]:>15.1f}%")
        elif fmt == "f2":
            print(f"  {name:<22} {vals[0]:>16.2f} {vals[1]:>16.2f} {vals[2]:>16.2f} {vals[3]:>16.2f}")
        elif fmt == "inr":
            print(f"  {name:<22} {'₹{:+,.0f}'.format(vals[0]):>16} {'₹{:+,.0f}'.format(vals[1]):>16} {'₹{:+,.0f}'.format(vals[2]):>16} {'₹{:+,.0f}'.format(vals[3]):>16}")

    row("Total Trades", "total_trades", "d")

    wins = [int(m["total_trades"] * m["win_rate"] / 100) for _, m, _ in ms]
    print(f"  {'Wins':<22} {wins[0]:>16} {wins[1]:>16} {wins[2]:>16} {wins[3]:>16}")

    row("Win Rate", "win_rate", "pct")
    row("Profit Factor", "profit_factor", "f2")
    row("Net P&L", "net_pnl", "inr")

    caps = [r["final_capital"] for _, _, r in ms]
    print(f"  {'Final Capital':<22} {'₹{:,.0f}'.format(caps[0]):>16} {'₹{:,.0f}'.format(caps[1]):>16} {'₹{:,.0f}'.format(caps[2]):>16} {'₹{:,.0f}'.format(caps[3]):>16}")

    row("Max Drawdown", "max_drawdown_pct", "pct")
    row("Sharpe Estimate", "sharpe_estimate", "f2")
    row("Avg Win", "avg_win_pnl", "inr")
    row("Avg Loss", "avg_loss_pnl", "inr")

    rrs = [abs(m["avg_win_pnl"] / m["avg_loss_pnl"]) if m["avg_loss_pnl"] != 0 else 0 for _, m, _ in ms]
    print(f"  {'Avg R:R':<22} {rrs[0]:>15.2f}x {rrs[1]:>15.2f}x {rrs[2]:>15.2f}x {rrs[3]:>15.2f}x")

    row("Max Consec Losses", "max_consecutive_losses", "d")

    blocked = [r.get("trend_blocked", 0) for _, _, r in ms]
    print(f"  {'Trend Blocked':<22} {blocked[0]:>16} {'N/A':>16} {blocked[2]:>16} {'N/A':>16}")


def main():
    print("=" * 74)
    print("  SENSEX INTRADAY BACKTEST — Opening Range Breakout (ORB)")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lots × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty/trade")
    print("  Opening Range: First 30 min (6 × 5-min candles)")
    print("  Entry: Breakout above/below OR | SL: Opposite OR boundary | Target: 1:2 R:R")
    print("  Trend: VWAP + OR position + EMA Slope (score >= 2 to trade)")
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
    print("  PART 1: REGULAR CANDLES — ORB")
    print(f"{'═' * 74}")
    reg_runs = run_candle_type("REGULAR", data_regular)

    # ═══════════════════════════════════════════════════════════
    # Run Heikin Ashi candles
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 74}")
    print("  PART 2: HEIKIN ASHI CANDLES — ORB")
    print(f"{'═' * 74}")
    ha_runs = run_candle_type("HEIKIN ASHI", data_ha)

    # ═══════════════════════════════════════════════════════════
    # Final head-to-head
    # ═══════════════════════════════════════════════════════════
    print_comparison_table(reg_runs, ha_runs)

    print(f"\n{'─' * 74}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print("  Past performance does not guarantee future results.")
    print(f"{'─' * 74}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "orb_backtest_results.json")
    save_data = {
        "run_date": datetime.now().isoformat(),
        "strategy": "Opening Range Breakout (ORB)",
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "lot_size": LOT_SIZE,
            "num_lots": NUM_LOTS,
            "qty_per_trade": LOT_SIZE * NUM_LOTS,
            "or_candles": 6,
            "or_duration_min": 30,
            "rr_ratio": 2.0,
            "trend_filter": "VWAP + OR position + EMA(20) Slope",
            "trend_threshold": 2,
            "max_trades_per_day": 4,
            "data_interval": "5min",
            "data_source": "real SENSEX 1-min resampled to 5-min",
            "trading_days": n_days,
        },
    }
    for candle_type, runs in [("regular", reg_runs), ("heikin_ashi", ha_runs)]:
        for label, run_data in runs.items():
            r = run_data["result"]
            key = f"{candle_type}_{label}"
            save_data[key] = {
                "weekly_pnl": run_data["weeks"],
                "metrics": r["metrics"],
                "final_capital": r["final_capital"],
                "trend_blocked": r.get("trend_blocked", 0),
            }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    return reg_runs["WITH TREND FILTER"]["result"]


if __name__ == "__main__":
    main()
