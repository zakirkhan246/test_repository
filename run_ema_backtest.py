#!/usr/bin/env python3
"""
Backtest runner for 20 EMA Channel strategy on real SENSEX data.

Strategy: Enter when price closes above/below the 20 EMA channel
(formed by EMA of opens and EMA of closes). Exit on channel crossback.
Re-enter opposite direction if next candle confirms.

Trend Filter: 3-factor composite score (VWAP + Opening Range + EMA Slope).
Only trade in the direction of the day's trend (score >= 2 or <= -2).

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
from strategies.ema_channel_strategy import EMAChannelStrategy
from backtester.channel_engine import ChannelEngine


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
        channel_exits = len(trade_df[trade_df["result"].isin(["WIN", "LOSS", "BREAKEVEN"])])
        time_exits = len(trade_df[trade_df["result"] == "TIME_EXIT"])
        print(f"\n{'─' * 74}")
        print(f"  EXIT BREAKDOWN — {label}")
        print(f"{'─' * 74}")
        print(f"  Channel exits: {channel_exits}  |  Time exits (EOD): {time_exits}")

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


def main():
    print("=" * 74)
    print("  SENSEX INTRADAY BACKTEST — 20 EMA Channel + Trend Filter")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lots × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty/trade")
    print("  Timeframe: 5-min candles | Channel exit | Re-entry on confirmation")
    print("  Trend: VWAP + Opening Range + EMA Slope (score >= 2 to trade)")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)

    # Load real data
    data = load_sensex_data()
    n_days = data["date"].nunique()
    print(f"\n  Data: {n_days} trading days of real SENSEX 5-min candles")
    print(f"  Range: {data['date'].min()} to {data['date'].max()}")

    # Prepare indicators (includes trend score)
    strategy = EMAChannelStrategy(ema_period=20)
    prepared = strategy.prepare(data)

    # ── Trend score distribution ──
    trading_candles = prepared[prepared["candle_idx"] >= 6]
    score_counts = trading_candles["trend_score"].value_counts().sort_index()
    print(f"\n{'─' * 74}")
    print("  TREND SCORE DISTRIBUTION (trading candles only)")
    print(f"{'─' * 74}")
    total_candles = len(trading_candles)
    for score, count in score_counts.items():
        pct = count / total_candles * 100
        label = "BEARISH" if score <= -2 else "BULLISH" if score >= 2 else "NEUTRAL"
        bar = "█" * int(pct / 2)
        print(f"  Score {score:+d}: {count:>6} ({pct:5.1f}%) {bar}  [{label}]")

    bullish_pct = len(trading_candles[trading_candles["trend_score"] >= 2]) / total_candles * 100
    bearish_pct = len(trading_candles[trading_candles["trend_score"] <= -2]) / total_candles * 100
    neutral_pct = 100 - bullish_pct - bearish_pct
    print(f"\n  Tradeable: {bullish_pct + bearish_pct:.1f}% (Bullish {bullish_pct:.1f}% | Bearish {bearish_pct:.1f}%) | Neutral: {neutral_pct:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Run 1: WITH trend filter
    # ═══════════════════════════════════════════════════════════
    engine_filtered = ChannelEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=4,
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
        use_trend_filter=True,
        trend_threshold=2,
    )
    result_filtered = engine_filtered.run(prepared)

    weeks_filtered = print_results(
        "WITH TREND FILTER", result_filtered, INITIAL_CAPITAL
    )

    # ═══════════════════════════════════════════════════════════
    # Run 2: WITHOUT trend filter (baseline comparison)
    # ═══════════════════════════════════════════════════════════
    engine_raw = ChannelEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=4,
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
        use_trend_filter=False,
    )
    result_raw = engine_raw.run(prepared)

    weeks_raw = print_results(
        "WITHOUT TREND FILTER (baseline)", result_raw, INITIAL_CAPITAL
    )

    # ═══════════════════════════════════════════════════════════
    # Side-by-side comparison
    # ═══════════════════════════════════════════════════════════
    mf = result_filtered["metrics"]
    mr = result_raw["metrics"]

    print(f"\n{'═' * 74}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'═' * 74}")
    print(f"  {'Metric':<25} {'With Filter':>18} {'Without Filter':>18}")
    print(f"  {'─' * 25} {'─' * 18} {'─' * 18}")
    print(f"  {'Total Trades':<25} {mf['total_trades']:>18} {mr['total_trades']:>18}")
    print(f"  {'Win Rate':<25} {mf['win_rate']:>17.1f}% {mr['win_rate']:>17.1f}%")
    print(f"  {'Profit Factor':<25} {mf['profit_factor']:>18.2f} {mr['profit_factor']:>18.2f}")
    print(f"  {'Net P&L':<25} {'₹{:+,.2f}'.format(mf['net_pnl']):>18} {'₹{:+,.2f}'.format(mr['net_pnl']):>18}")
    print(f"  {'Final Capital':<25} {'₹{:,.2f}'.format(result_filtered['final_capital']):>18} {'₹{:,.2f}'.format(result_raw['final_capital']):>18}")
    print(f"  {'Max Drawdown':<25} {mf['max_drawdown_pct']:>17.2f}% {mr['max_drawdown_pct']:>17.2f}%")
    print(f"  {'Sharpe Estimate':<25} {mf['sharpe_estimate']:>18.2f} {mr['sharpe_estimate']:>18.2f}")
    print(f"  {'Avg Win':<25} {'₹{:+,.2f}'.format(mf['avg_win_pnl']):>18} {'₹{:+,.2f}'.format(mr['avg_win_pnl']):>18}")
    print(f"  {'Avg Loss':<25} {'₹{:+,.2f}'.format(mf['avg_loss_pnl']):>18} {'₹{:+,.2f}'.format(mr['avg_loss_pnl']):>18}")
    print(f"  {'Trend Blocked':<25} {result_filtered.get('trend_blocked', 0):>18} {'N/A':>18}")

    print(f"\n{'─' * 74}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print("  Past performance does not guarantee future results.")
    print(f"{'─' * 74}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "ema_backtest_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "strategy": "20 EMA Channel + Trend Filter",
            "config": {
                "initial_capital": INITIAL_CAPITAL,
                "lot_size": LOT_SIZE,
                "num_lots": NUM_LOTS,
                "qty_per_trade": LOT_SIZE * NUM_LOTS,
                "ema_period": 20,
                "trend_filter": "VWAP + Opening Range + EMA Channel Slope",
                "trend_threshold": 2,
                "max_trades_per_day": 4,
                "data_interval": "5min",
                "data_source": "real SENSEX 1-min resampled to 5-min",
                "trading_days": n_days,
            },
            "with_trend_filter": {
                "weekly_pnl": weeks_filtered,
                "metrics": mf,
                "final_capital": result_filtered["final_capital"],
                "trend_blocked": result_filtered.get("trend_blocked", 0),
            },
            "without_trend_filter": {
                "weekly_pnl": weeks_raw,
                "metrics": mr,
                "final_capital": result_raw["final_capital"],
            },
        }, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    return result_filtered


if __name__ == "__main__":
    main()
