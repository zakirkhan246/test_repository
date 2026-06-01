#!/usr/bin/env python3
"""
Backtest runner for VWAP Bounce strategy on real SENSEX 1-min data.

Uses only trading days with real volume data (Nov 2025 - Mar 2026).
Capital: 20,000 INR | Lot size: 20 (1 SENSEX lot) | 1-min candles
"""

import sys
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.vwap_bounce_strategy import compute_vwap, detect_vwap_bounce_signals
from backtester.engine import BacktestEngine
from backtester.metrics import compute_metrics


INITIAL_CAPITAL = 20_000.0
LOT_SIZE = 20
NUM_LOTS = 1


def load_1min_with_volume(csv_path=None):
    """Load 1-min data, keep only days with real volume."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "sensex_1min_2yr.csv")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.set_index("datetime", inplace=True)
    df["date"] = df.index.date

    # Keep only days where volume exists
    daily_vol = df.groupby("date")["volume"].sum()
    vol_days = daily_vol[daily_vol > 0].index
    df = df[df["date"].isin(vol_days)].copy()

    return df


def weekly_pnl_table(trades, initial_capital):
    if not trades:
        return []

    tdf = pd.DataFrame(trades)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf["week_start"] = tdf["entry_time"].dt.to_period("W").apply(lambda p: p.start_time)

    rows = []
    running_capital = initial_capital

    for week, group in tdf.groupby("week_start"):
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


def print_results(result, initial_capital):
    trades = result["trades"]
    m = result["metrics"]
    weeks = weekly_pnl_table(trades, initial_capital)

    print(f"\n{'─' * 74}")
    print(f"  WEEKLY P&L — VWAP Bounce (1-min)")
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
        print(f"  EXIT BREAKDOWN")
        print(f"{'─' * 74}")
        print(f"  Target hits (WIN): {target_exits}  |  SL hits (LOSS): {sl_exits}  |  Time exits (EOD): {time_exits}")

    print(f"\n{'─' * 74}")
    print(f"  SUMMARY — VWAP Bounce (1-min)")
    print(f"{'─' * 74}")
    print(f"  Starting Capital:  ₹{initial_capital:>10,.2f}")
    print(f"  Final Capital:     ₹{result['final_capital']:>10,.2f}")
    print(f"  Net P&L:           ₹{m['net_pnl']:>+10,.2f}  ({m['net_pnl_pct']:+.2f}%)")
    print(f"  Total Trades:      {m['total_trades']}")
    print(f"  Win Rate:          {m['win_rate']:.1f}%")
    print(f"  Profit Factor:     {m['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Estimate:   {m['sharpe_estimate']:.2f}")
    print(f"\n  Avg Win:           ₹{m['avg_win_pnl']:+,.2f}  ({m['avg_win_duration_candles']:.1f} candles / {m['avg_win_duration_candles'] * 1:.0f} min)")
    print(f"  Avg Loss:          ₹{m['avg_loss_pnl']:+,.2f}  ({m['avg_loss_duration_candles']:.1f} candles / {m['avg_loss_duration_candles'] * 1:.0f} min)")
    print(f"\n  Long Trades:       {m['long_trades']}  (Win: {m['long_win_rate']:.1f}%)")
    print(f"  Short Trades:      {m['short_trades']}  (Win: {m['short_win_rate']:.1f}%)")
    print(f"  Max Consec Wins:   {m['max_consecutive_wins']}")
    print(f"  Max Consec Losses: {m['max_consecutive_losses']}")

    return weeks


def main():
    print("=" * 74)
    print("  SENSEX INTRADAY BACKTEST — VWAP Bounce Strategy")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lot × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty/trade")
    print("  Timeframe: 1-minute candles")
    print("  Data: Only days with real volume (Nov 2025 - Mar 2026)")
    print("  Window: 10:15 - 15:00 (skip first hour + post 3pm)")
    print("  Entry: Bounce off VWAP | SL: Bounce candle extreme | Target: Last swing")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    # Load data
    df = load_1min_with_volume()
    n_days = df["date"].nunique()
    print(f"\n  Data: {n_days} trading days, {len(df)} 1-min candles")
    print(f"  Range: {df['date'].min()} to {df['date'].max()}")

    # Candle index within day
    df["candle_idx"] = df.groupby("date").cumcount()

    # Compute VWAP
    df["vwap"] = compute_vwap(df)
    print(f"  VWAP computed for all candles")

    # Detect signals — skip first hour (candles 0-59) and post 3pm (candle 345+)
    # 9:15 + 60 min = 10:15 (candle 60), 9:15 + 345 min = 15:00 (candle 345)
    df_signals = detect_vwap_bounce_signals(
        df, lookback=60, min_rr=1.0, start_candle=60, end_candle=344,
    )

    total_signals = (df_signals["signal"] != 0).sum()
    long_signals = (df_signals["signal"] == 1).sum()
    short_signals = (df_signals["signal"] == -1).sum()
    print(f"\n  Signals: {total_signals} total ({long_signals} long, {short_signals} short)")

    signal_days = df_signals[df_signals["signal"] != 0]["date"].nunique()
    print(f"  Signal days: {signal_days} / {n_days} ({signal_days / n_days * 100:.1f}%)")

    # Run backtest
    # force_close_candle = 370 (for 375 candles/day on 1-min, close ~5 min before end)
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=10,
        force_close_candle=370,
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
    )
    result = engine.run(df_signals)
    weeks = print_results(result, INITIAL_CAPITAL)

    # Trade details sample
    if result["trades"]:
        print(f"\n{'─' * 74}")
        print(f"  SAMPLE TRADES (first 15)")
        print(f"{'─' * 74}")
        sample = result["trades"][:15]
        sample_table = []
        for t in sample:
            direction = "LONG" if t["direction"] == 1 else "SHORT"
            sample_table.append([
                str(t["entry_time"])[:16],
                direction,
                f"{t['entry_price']:.0f}",
                f"{t['stop_loss']:.0f}",
                f"{t['target']:.0f}",
                f"{t['exit_price']:.0f}",
                f"₹{t['pnl']:+,.0f}",
                t["result"],
                f"{t['duration_candles']}m",
            ])
        headers = ["Entry", "Dir", "Price", "SL", "Tgt", "Exit", "P&L", "Result", "Dur"]
        print(tabulate(sample_table, headers=headers, tablefmt="simple", stralign="right"))

    print(f"\n{'─' * 74}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print(f"{'─' * 74}\n")

    return result


if __name__ == "__main__":
    main()
