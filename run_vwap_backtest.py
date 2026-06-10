#!/usr/bin/env python3
"""
Backtest runner for VWAP Bounce strategy on real SENSEX 1-min data.

Final strategy with all filters:
  - Time window: 10:15 - 15:00
  - ADX > 20 (trending market)
  - Choppiness Index < 50 (not choppy)
  - Candle colour confirmation (green for long, red for short)
  - Volume cap < 2x day average
  - VWAP congestion filter: skip if >= 5 of last 10 candles straddle VWAP
  - Reversal exit: close trade when MFE >= 70% of target and candle closes against prev

Capital: ₹20,000 | 1 SENSEX lot (20 qty) | 1-min candles
Data: Nov 2025 - Mar 2026 (82 days with real volume)
"""

import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.vwap_bounce_strategy import compute_vwap, detect_vwap_bounce_signals
from strategies.indicators import compute_adx, compute_choppiness_index
from backtester.engine import BacktestEngine


INITIAL_CAPITAL = 20_000.0
LOT_SIZE = 20
NUM_LOTS = 1

START_CANDLE = 60   # skip first hour (10:15)
END_CANDLE = 344    # no entries post 3:00pm

ADX_THRESHOLD = 20
CHOP_THRESHOLD = 50
MAX_VOL_RATIO = 2.0
MAX_VWAP_TOUCHES = 5
REVERSAL_EXIT_PCT = 70.0


def load_1min_with_volume(csv_path=None):
    """Load 1-min data, keep only days with real volume."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "sensex_1min_2yr.csv")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.set_index("datetime", inplace=True)
    df["date"] = df.index.date

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


def apply_chop_filter(df_signals, adx_thresh, chop_thresh):
    """Zero out signals where ADX <= threshold OR Choppiness >= threshold."""
    filtered = df_signals.copy()
    choppy = (filtered["adx"] <= adx_thresh) | (filtered["chop"] >= chop_thresh)
    filtered.loc[choppy, "signal"] = 0
    filtered.loc[choppy, "stop_loss"] = np.nan
    filtered.loc[choppy, "target"] = np.nan
    return filtered


def main():
    print("=" * 80)
    print("  SENSEX VWAP BOUNCE — Final Strategy Backtest")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lot × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty")
    print("  1-min candles | Entry window: 10:15 - 15:00")
    print(f"  Filters: ADX>{ADX_THRESHOLD} + CI<{CHOP_THRESHOLD} + Vol<{MAX_VOL_RATIO}x + VWAPtouches<{MAX_VWAP_TOUCHES}")
    print(f"  Reversal exit: close-vs-prev at {REVERSAL_EXIT_PCT:.0f}% MFE")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data
    df = load_1min_with_volume()
    n_days = df["date"].nunique()
    print(f"\n  Data: {n_days} trading days, {len(df)} 1-min candles")
    print(f"  Range: {df['date'].min()} to {df['date'].max()}")

    # Prep: candle index, VWAP
    df["candle_idx"] = df.groupby("date").cumcount()
    df["vwap"] = compute_vwap(df)

    # ADX and CI computed per day to avoid overnight gap contamination
    df["adx"] = np.nan
    df["chop"] = np.nan
    for date in df["date"].unique():
        mask = df["date"] == date
        day = df.loc[mask]
        if len(day) < 30:
            continue
        adx_df = compute_adx(day["high"], day["low"], day["close"], period=14)
        df.loc[mask, "adx"] = adx_df["adx"].values
        df.loc[mask, "chop"] = compute_choppiness_index(
            day["high"], day["low"], day["close"], period=14
        ).values

    print(f"  ADX range: {df['adx'].min():.1f} - {df['adx'].max():.1f} (mean: {df['adx'].mean():.1f})")
    print(f"  Chop range: {df['chop'].min():.1f} - {df['chop'].max():.1f} (mean: {df['chop'].mean():.1f})")

    # Generate signals with volume cap + VWAP congestion filter built in
    df_signals = detect_vwap_bounce_signals(
        df,
        lookback=60,
        min_rr=1.0,
        start_candle=START_CANDLE,
        end_candle=END_CANDLE,
        max_vol_ratio=MAX_VOL_RATIO,
        max_vwap_touches=MAX_VWAP_TOUCHES,
    )

    raw_sigs = (df_signals["signal"] != 0).sum()
    print(f"\n  Signals after vol/congestion filter: {raw_sigs}")

    # Apply ADX + Choppiness filter
    df_filtered = apply_chop_filter(df_signals, ADX_THRESHOLD, CHOP_THRESHOLD)
    final_sigs = (df_filtered["signal"] != 0).sum()
    print(f"  Signals after chop filter (ADX>{ADX_THRESHOLD} + CI<{CHOP_THRESHOLD}): {final_sigs}")

    # Run backtest with reversal exit
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=10,
        force_close_candle=360,  # 15:15
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
        reversal_exit_pct=REVERSAL_EXIT_PCT,
    )
    result = engine.run(df_filtered)

    trades = result["trades"]
    m = result["metrics"]

    # Weekly P&L table
    weeks = weekly_pnl_table(trades, INITIAL_CAPITAL)

    print(f"\n{'─' * 74}")
    print(f"  WEEKLY P&L")
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

    # Trade breakdown
    if trades:
        tdf = pd.DataFrame(trades)
        target_hits = len(tdf[tdf["result"] == "WIN"])
        sl_hits = len(tdf[tdf["result"] == "LOSS"])
        time_exits = len(tdf[tdf["result"] == "TIME_EXIT"])
        rev_exits = len(tdf[tdf["result"].isin(["WIN", "LOSS", "BREAKEVEN"]) & (tdf["exit_price"] != tdf["stop_loss"]) & (tdf["exit_price"] != tdf["target"])])
        print(f"\n  Target hits: {target_hits} | SL hits: {sl_hits} | Time exits: {time_exits}")

    # Summary
    print(f"\n{'═' * 74}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'═' * 74}")
    print(f"  Trades:     {m['total_trades']}")
    print(f"  Win Rate:   {m['win_rate']:.1f}%")
    print(f"  Profit Factor: {m['profit_factor']:.2f}")
    print(f"  Net P&L:    ₹{m['net_pnl']:+,.2f} ({m['net_pnl_pct']:+.2f}%)")
    print(f"  Final Capital: ₹{result['final_capital']:,.2f}")
    print(f"  Max Drawdown:  {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe:     {m['sharpe_estimate']:.2f}")
    print(f"  Avg Win:    ₹{m['avg_win_pnl']:+,.2f} ({m['avg_win_duration_candles']:.0f} candles)")
    print(f"  Avg Loss:   ₹{m['avg_loss_pnl']:+,.2f} ({m['avg_loss_duration_candles']:.0f} candles)")

    avg_rr = abs(m["avg_win_pnl"] / m["avg_loss_pnl"]) if m["avg_loss_pnl"] != 0 else 0
    print(f"  Avg R:R:    {avg_rr:.2f}x")
    print(f"  Long:  {m['long_trades']} trades ({m['long_win_rate']:.1f}% WR)")
    print(f"  Short: {m['short_trades']} trades ({m['short_win_rate']:.1f}% WR)")
    print(f"  Max Consec Wins: {m['max_consecutive_wins']} | Losses: {m['max_consecutive_losses']}")

    print(f"\n{'─' * 74}")
    print("  RISK DISCLAIMER: Educational/research only. Past performance does not")
    print("  guarantee future results. This is not financial advice.")
    print(f"{'─' * 74}\n")


if __name__ == "__main__":
    main()
