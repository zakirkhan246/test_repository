#!/usr/bin/env python3
"""
Backtest runner for 20 EMA Channel strategy on real SENSEX data.

Runs on the same 82 volume days (Nov 2025 - Mar 2026) as the VWAP Bounce
strategy for a fair head-to-head comparison.

Tests 4 variants:
  1. Trend filter only (baseline on volume days)
  2. Trend filter + ADX > 20
  3. Trend filter + CI < 50
  4. Trend filter + ADX > 20 + CI < 50

Capital: ₹20,000 | Lot size: 10 × 2 = 20 qty | 5-min candles
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
from strategies.ema_channel_strategy import EMAChannelStrategy
from strategies.indicators import compute_adx, compute_choppiness_index
from backtester.channel_engine import ChannelEngine


INITIAL_CAPITAL = 20_000.0
LOT_SIZE = 10
NUM_LOTS = 2


def load_volume_days_only(heikin_ashi=False):
    """Load 5-min data, keep only days with real volume (same as VWAP strategy)."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "sensex_1min_2yr.csv")
    df_1min = pd.read_csv(csv_path, parse_dates=["datetime"])
    df_1min = df_1min.sort_values("datetime").reset_index(drop=True)
    df_1min = df_1min.dropna(subset=["open", "high", "low", "close"])
    df_1min.set_index("datetime", inplace=True)
    df_1min["date"] = df_1min.index.date

    daily_vol = df_1min.groupby("date")["volume"].sum()
    vol_days = daily_vol[daily_vol > 0].index

    df_1min = df_1min[df_1min["date"].isin(vol_days)].copy()

    resampled_frames = []
    for date, day_df in df_1min.groupby("date"):
        day_resampled = day_df.resample("5min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
        resampled_frames.append(day_resampled)

    df = pd.concat(resampled_frames)
    df["date"] = df.index.date
    df["symbol"] = "SENSEX"

    if heikin_ashi:
        from data.loader import convert_to_heikin_ashi
        df = convert_to_heikin_ashi(df)

    return df


def add_chop_indicators(df):
    """Compute ADX and CI per day on 5-min candles."""
    df["adx"] = np.nan
    df["chop"] = np.nan
    for date in df["date"].unique():
        mask = df["date"] == date
        day = df.loc[mask]
        if len(day) < 15:
            continue
        adx_df = compute_adx(day["high"], day["low"], day["close"], period=14)
        df.loc[mask, "adx"] = adx_df["adx"].values
        df.loc[mask, "chop"] = compute_choppiness_index(
            day["high"], day["low"], day["close"], period=14
        ).values
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

        rows.append({
            "week": week.strftime("%d-%b-%Y"),
            "days": group["date"].nunique(),
            "trades": len(group),
            "wins": wins,
            "losses": losses,
            "win_rate": f"{wins / len(group) * 100:.0f}%" if len(group) > 0 else "0%",
            "pnl": round(week_pnl, 2),
            "capital": round(running_capital, 2),
        })

    return rows


def print_results(label, result, initial_capital):
    trades = result["trades"]
    m = result["metrics"]
    weeks = weekly_pnl_table(trades, initial_capital)

    print(f"\n{'─' * 74}")
    print(f"  WEEKLY P&L — {label}")
    print(f"{'─' * 74}\n")

    table = []
    for w in weeks:
        table.append([
            w["week"], w["days"], w["trades"], w["wins"], w["losses"],
            w["win_rate"], f"₹{w['pnl']:+,.2f}", f"₹{w['capital']:,.2f}",
        ])
    headers = ["Week Of", "Days", "Trades", "W", "L", "Win%", "P&L", "Capital"]
    print(tabulate(table, headers=headers, tablefmt="simple", stralign="right"))

    if weeks:
        best_w = max(weeks, key=lambda w: w["pnl"])
        worst_w = min(weeks, key=lambda w: w["pnl"])
        green = sum(1 for w in weeks if w["pnl"] > 0)
        red = sum(1 for w in weeks if w["pnl"] <= 0)
        print(f"\n  Green weeks: {green} | Red weeks: {red}")
        print(f"  Best week:  {best_w['week']}  ₹{best_w['pnl']:+,.2f}")
        print(f"  Worst week: {worst_w['week']}  ₹{worst_w['pnl']:+,.2f}")

    print(f"\n  Net P&L: ₹{m['net_pnl']:+,.2f} ({m['net_pnl_pct']:+.2f}%) | Trades: {m['total_trades']} | WR: {m['win_rate']:.1f}%")
    print(f"  PF: {m['profit_factor']:.2f} | MaxDD: {m['max_drawdown_pct']:.2f}% | Sharpe: {m['sharpe_estimate']:.2f}")
    print(f"  Avg Win: ₹{m['avg_win_pnl']:+,.2f} | Avg Loss: ₹{m['avg_loss_pnl']:+,.2f}")
    rr = abs(m["avg_win_pnl"] / m["avg_loss_pnl"]) if m["avg_loss_pnl"] != 0 else 0
    print(f"  Avg R:R: {rr:.2f}x")
    print(f"  Long: {m['long_trades']} ({m['long_win_rate']:.1f}%) | Short: {m['short_trades']} ({m['short_win_rate']:.1f}%)")
    print(f"  Max Consec Wins: {m['max_consecutive_wins']} | Losses: {m['max_consecutive_losses']}")
    if result.get("trend_blocked", 0) > 0:
        print(f"  Trend/chop blocked: {result['trend_blocked']} entries")

    return weeks


def main():
    print("=" * 80)
    print("  SENSEX EMA CHANNEL — Volume Days Only + Chop Filter Comparison")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lots × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty")
    print("  5-min candles | Heikin Ashi | Trend filter ON")
    print("  Volume days only (same 82 days as VWAP Bounce)")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load volume days only, Heikin Ashi (best variant from prior testing)
    df = load_volume_days_only(heikin_ashi=True)
    n_days = df["date"].nunique()
    print(f"\n  Data: {n_days} trading days, {len(df)} 5-min candles")
    print(f"  Range: {df['date'].min()} to {df['date'].max()}")

    # Prepare strategy indicators
    strategy = EMAChannelStrategy(ema_period=20)
    prepared = strategy.prepare(df)

    # Add ADX and CI per day
    prepared = add_chop_indicators(prepared)
    print(f"  ADX range: {prepared['adx'].min():.1f} - {prepared['adx'].max():.1f} (mean: {prepared['adx'].mean():.1f})")
    print(f"  Chop range: {prepared['chop'].min():.1f} - {prepared['chop'].max():.1f} (mean: {prepared['chop'].mean():.1f})")

    # Run 4 variants
    variants = [
        ("Trend Only (baseline)", 0.0, 100.0),
        ("Trend + ADX>20", 20.0, 100.0),
        ("Trend + CI<50", 0.0, 50.0),
        ("Trend + ADX>20 + CI<50", 20.0, 50.0),
    ]

    results = []
    for label, adx_thresh, chop_thresh in variants:
        engine = ChannelEngine(
            initial_capital=INITIAL_CAPITAL,
            max_trades_per_day=4,
            lot_size=LOT_SIZE,
            num_lots=NUM_LOTS,
            use_trend_filter=True,
            trend_threshold=2,
            adx_threshold=adx_thresh,
            chop_threshold=chop_thresh,
        )
        result = engine.run(prepared)
        results.append((label, result))

    # Print comparison table
    print(f"\n{'═' * 90}")
    print("  CHOP FILTER COMPARISON — EMA Channel (Heikin Ashi + Trend Filter)")
    print(f"{'═' * 90}")

    scan_table = []
    for label, r in results:
        m = r["metrics"]
        rr = abs(m["avg_win_pnl"] / m["avg_loss_pnl"]) if m["avg_loss_pnl"] != 0 else 0
        scan_table.append([
            label, m["total_trades"], f"{m['win_rate']:.1f}%",
            f"{m['profit_factor']:.2f}", f"₹{m['net_pnl']:+,.0f}",
            f"{m['max_drawdown_pct']:.1f}%", f"{m['sharpe_estimate']:.2f}",
            f"{rr:.1f}x", m["max_consecutive_losses"],
        ])
    headers = ["Variant", "Trades", "WR", "PF", "Net P&L", "MaxDD", "Sharpe", "R:R", "MaxLStreak"]
    print(tabulate(scan_table, headers=headers, tablefmt="simple", stralign="right"))

    # Print full weekly results for best variant
    best_label, best_result = max(results, key=lambda r: r[1]["metrics"]["profit_factor"])
    print(f"\n  Best by PF: {best_label}")
    print_results(best_label, best_result, INITIAL_CAPITAL)

    # Also print baseline for reference
    base_label, base_result = results[0]
    if base_label != best_label:
        print_results(base_label, base_result, INITIAL_CAPITAL)

    print(f"\n{'─' * 80}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print("  Past performance does not guarantee future results.")
    print(f"{'─' * 80}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "ema_backtest_results.json")
    save_data = {
        "run_date": datetime.now().isoformat(),
        "strategy": "20 EMA Channel + Trend Filter + Chop Filter",
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "lot_size": LOT_SIZE,
            "num_lots": NUM_LOTS,
            "qty_per_trade": LOT_SIZE * NUM_LOTS,
            "ema_period": 20,
            "candle_type": "Heikin Ashi",
            "data_interval": "5min",
            "volume_days_only": True,
            "trading_days": n_days,
        },
    }
    for label, r in results:
        key = label.replace(" ", "_").replace("+", "").replace(">", "gt").replace("<", "lt")
        save_data[key] = {
            "metrics": r["metrics"],
            "final_capital": r["final_capital"],
        }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
