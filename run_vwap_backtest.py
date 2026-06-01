#!/usr/bin/env python3
"""
Backtest runner for VWAP Bounce strategy on real SENSEX 1-min data.

Tests three variants side-by-side:
  1. No filter (baseline)
  2. ADX filter — only trade when ADX > threshold (trending)
  3. Choppiness Index filter — only trade when CI < threshold (trending)

Uses only trading days with real volume data (Nov 2025 - Mar 2026).
Capital: 20,000 INR | Lot size: 20 (1 SENSEX lot) | 1-min candles
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
END_CANDLE = 344    # no entries post 3pm


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


def run_backtest(df_signals, label):
    """Run backtest and return result dict."""
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        max_trades_per_day=10,
        force_close_candle=370,
        lot_size=LOT_SIZE,
        num_lots=NUM_LOTS,
    )
    result = engine.run(df_signals)
    return result


def print_full_results(label, result):
    """Print weekly P&L and summary for a single run."""
    trades = result["trades"]
    m = result["metrics"]
    weeks = weekly_pnl_table(trades, INITIAL_CAPITAL)

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

    if trades:
        tdf = pd.DataFrame(trades)
        print(f"\n  Target: {len(tdf[tdf['result']=='WIN'])} | SL: {len(tdf[tdf['result']=='LOSS'])} | EOD: {len(tdf[tdf['result']=='TIME_EXIT'])}")

    print(f"\n  Net P&L: ₹{m['net_pnl']:+,.2f} ({m['net_pnl_pct']:+.2f}%) | Trades: {m['total_trades']} | WR: {m['win_rate']:.1f}%")
    print(f"  PF: {m['profit_factor']:.2f} | MaxDD: {m['max_drawdown_pct']:.2f}% | Sharpe: {m['sharpe_estimate']:.2f}")
    print(f"  Avg Win: ₹{m['avg_win_pnl']:+,.2f} | Avg Loss: ₹{m['avg_loss_pnl']:+,.2f}")
    print(f"  Long: {m['long_trades']} ({m['long_win_rate']:.1f}%) | Short: {m['short_trades']} ({m['short_win_rate']:.1f}%)")
    print(f"  Max Consec Wins: {m['max_consecutive_wins']} | Losses: {m['max_consecutive_losses']}")

    return weeks


def print_comparison(results):
    """Side-by-side comparison table."""
    print(f"\n{'═' * 80}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'═' * 80}")

    labels = [r[0] for r in results]
    metrics = [r[1]["metrics"] for r in results]
    finals = [r[1]["final_capital"] for r in results]

    header = f"  {'Metric':<24}" + "".join(f"{l:>18}" for l in labels)
    print(header)
    print(f"  {'─' * 24}" + "─" * 18 * len(labels))

    def row(name, key, fmt="d"):
        vals = [m[key] for m in metrics]
        line = f"  {name:<24}"
        for v in vals:
            if fmt == "d":
                line += f"{v:>18}"
            elif fmt == "pct":
                line += f"{v:>17.1f}%"
            elif fmt == "f2":
                line += f"{v:>18.2f}"
            elif fmt == "inr":
                line += f"{'₹{:+,.0f}'.format(v):>18}"
        print(line)

    row("Total Trades", "total_trades", "d")
    row("Win Rate", "win_rate", "pct")
    row("Profit Factor", "profit_factor", "f2")
    row("Net P&L", "net_pnl", "inr")

    line = f"  {'Final Capital':<24}"
    for f in finals:
        line += f"{'₹{:,.0f}'.format(f):>18}"
    print(line)

    row("Max Drawdown", "max_drawdown_pct", "pct")
    row("Sharpe Estimate", "sharpe_estimate", "f2")
    row("Avg Win", "avg_win_pnl", "inr")
    row("Avg Loss", "avg_loss_pnl", "inr")

    # Avg R:R
    line = f"  {'Avg R:R':<24}"
    for m in metrics:
        rr = abs(m["avg_win_pnl"] / m["avg_loss_pnl"]) if m["avg_loss_pnl"] != 0 else 0
        line += f"{rr:>17.2f}x"
    print(line)

    row("Max Consec Wins", "max_consecutive_wins", "d")
    row("Max Consec Losses", "max_consecutive_losses", "d")
    row("Long Win Rate", "long_win_rate", "pct")
    row("Short Win Rate", "short_win_rate", "pct")

    # Highlight winner
    best_pf = max(results, key=lambda r: r[1]["metrics"]["profit_factor"])
    best_pnl = max(results, key=lambda r: r[1]["metrics"]["net_pnl"])
    best_dd = min(results, key=lambda r: r[1]["metrics"]["max_drawdown_pct"])
    print(f"\n  Best PF: {best_pf[0]} | Best P&L: {best_pnl[0]} | Lowest DD: {best_dd[0]}")


def apply_filter(df_signals, filter_col, filter_op, threshold):
    """Zero out signals where filter condition is not met."""
    filtered = df_signals.copy()
    if filter_op == ">":
        mask = filtered[filter_col] <= threshold
    else:  # "<"
        mask = filtered[filter_col] >= threshold
    filtered.loc[mask, "signal"] = 0
    filtered.loc[mask, "stop_loss"] = np.nan
    filtered.loc[mask, "target"] = np.nan
    return filtered


def main():
    print("=" * 80)
    print("  SENSEX VWAP BOUNCE — Chop Filter Comparison")
    print(f"  Capital: ₹{INITIAL_CAPITAL:,.0f} | {NUM_LOTS} lot × {LOT_SIZE} = {LOT_SIZE * NUM_LOTS} qty")
    print("  1-min candles | Window: 10:15 - 15:00")
    print("  Filters: No filter vs ADX vs Choppiness Index")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data
    df = load_1min_with_volume()
    n_days = df["date"].nunique()
    print(f"\n  Data: {n_days} trading days, {len(df)} 1-min candles")
    print(f"  Range: {df['date'].min()} to {df['date'].max()}")

    # Prep: candle index, VWAP, ADX, Choppiness
    df["candle_idx"] = df.groupby("date").cumcount()
    df["vwap"] = compute_vwap(df)

    adx_df = compute_adx(df["high"], df["low"], df["close"], period=14)
    df["adx"] = adx_df["adx"]

    df["chop"] = compute_choppiness_index(df["high"], df["low"], df["close"], period=14)

    print(f"  ADX range: {df['adx'].min():.1f} - {df['adx'].max():.1f} (mean: {df['adx'].mean():.1f})")
    print(f"  Chop range: {df['chop'].min():.1f} - {df['chop'].max():.1f} (mean: {df['chop'].mean():.1f})")

    # Generate base signals
    df_signals = detect_vwap_bounce_signals(
        df, lookback=60, min_rr=1.0, start_candle=START_CANDLE, end_candle=END_CANDLE,
    )

    total_sigs = (df_signals["signal"] != 0).sum()
    print(f"\n  Base signals: {total_sigs}")

    # ═══════════════════════════════════════════════════════════
    #  1. BASELINE — No filter
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("  RUN 1: NO FILTER (Baseline)")
    print(f"{'═' * 80}")
    result_base = run_backtest(df_signals, "No Filter")
    weeks_base = print_full_results("No Filter", result_base)

    # ═══════════════════════════════════════════════════════════
    #  2. ADX FILTER — Test multiple thresholds
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("  ADX THRESHOLD SCAN")
    print(f"{'═' * 80}")

    adx_thresholds = [15, 20, 25, 30]
    adx_results = {}

    scan_table = []
    for thresh in adx_thresholds:
        filtered = apply_filter(df_signals, "adx", ">", thresh)
        sigs = (filtered["signal"] != 0).sum()
        r = run_backtest(filtered, f"ADX>{thresh}")
        m = r["metrics"]
        adx_results[thresh] = r
        scan_table.append([
            f"ADX > {thresh}", sigs, m["total_trades"], f"{m['win_rate']:.1f}%",
            f"{m['profit_factor']:.2f}", f"₹{m['net_pnl']:+,.0f}",
            f"{m['max_drawdown_pct']:.1f}%", m["max_consecutive_losses"],
        ])

    headers = ["Filter", "Sigs", "Trades", "WR", "PF", "Net P&L", "MaxDD", "MaxLStreak"]
    print(tabulate(scan_table, headers=headers, tablefmt="simple", stralign="right"))

    # Pick best ADX
    best_adx_thresh = max(adx_results, key=lambda t: adx_results[t]["metrics"]["profit_factor"])
    print(f"\n  Best ADX threshold: > {best_adx_thresh} (PF: {adx_results[best_adx_thresh]['metrics']['profit_factor']:.2f})")

    # Full results for best ADX
    best_adx_filtered = apply_filter(df_signals, "adx", ">", best_adx_thresh)
    result_adx = run_backtest(best_adx_filtered, f"ADX > {best_adx_thresh}")
    weeks_adx = print_full_results(f"ADX > {best_adx_thresh}", result_adx)

    # ═══════════════════════════════════════════════════════════
    #  3. CHOPPINESS INDEX FILTER — Test multiple thresholds
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("  CHOPPINESS INDEX THRESHOLD SCAN")
    print(f"{'═' * 80}")

    chop_thresholds = [50, 55, 61.8, 65]
    chop_results = {}

    scan_table = []
    for thresh in chop_thresholds:
        filtered = apply_filter(df_signals, "chop", "<", thresh)
        sigs = (filtered["signal"] != 0).sum()
        r = run_backtest(filtered, f"CI<{thresh}")
        m = r["metrics"]
        chop_results[thresh] = r
        scan_table.append([
            f"CI < {thresh}", sigs, m["total_trades"], f"{m['win_rate']:.1f}%",
            f"{m['profit_factor']:.2f}", f"₹{m['net_pnl']:+,.0f}",
            f"{m['max_drawdown_pct']:.1f}%", m["max_consecutive_losses"],
        ])

    print(tabulate(scan_table, headers=headers, tablefmt="simple", stralign="right"))

    # Pick best Chop
    best_chop_thresh = max(chop_results, key=lambda t: chop_results[t]["metrics"]["profit_factor"])
    print(f"\n  Best Choppiness threshold: < {best_chop_thresh} (PF: {chop_results[best_chop_thresh]['metrics']['profit_factor']:.2f})")

    # Full results for best Chop
    best_chop_filtered = apply_filter(df_signals, "chop", "<", best_chop_thresh)
    result_chop = run_backtest(best_chop_filtered, f"CI < {best_chop_thresh}")
    weeks_chop = print_full_results(f"CI < {best_chop_thresh}", result_chop)

    # ═══════════════════════════════════════════════════════════
    #  HEAD-TO-HEAD
    # ═══════════════════════════════════════════════════════════
    print_comparison([
        ("No Filter", result_base),
        (f"ADX>{best_adx_thresh}", result_adx),
        (f"CI<{best_chop_thresh}", result_chop),
    ])

    # ═══════════════════════════════════════════════════════════
    #  FILTER IMPACT — wins cut vs losses cut
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 80}")
    print("  FILTER IMPACT — Wins Cut vs Losses Cut")
    print(f"{'═' * 80}")

    base_trades = result_base["trades"]
    base_wins = sum(1 for t in base_trades if t["result"] == "WIN")
    base_losses = len(base_trades) - base_wins

    for label, result in [(f"ADX>{best_adx_thresh}", result_adx), (f"CI<{best_chop_thresh}", result_chop)]:
        filt_trades = result["trades"]
        filt_wins = sum(1 for t in filt_trades if t["result"] == "WIN")
        filt_losses = len(filt_trades) - filt_wins
        wins_cut = base_wins - filt_wins
        losses_cut = base_losses - filt_losses
        print(f"\n  {label}:")
        print(f"    Trades: {len(base_trades)} → {len(filt_trades)} (cut {len(base_trades)-len(filt_trades)})")
        print(f"    Wins cut:   {wins_cut} (of {base_wins})")
        print(f"    Losses cut: {losses_cut} (of {base_losses})")
        if wins_cut > 0:
            print(f"    Loss:Win cut ratio: {losses_cut/wins_cut:.1f}:1 (higher = better filter)")

    print(f"\n{'─' * 80}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print(f"{'─' * 80}\n")


if __name__ == "__main__":
    main()
