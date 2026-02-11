#!/usr/bin/env python3
"""
Main backtest runner for NIFTY 50 & SENSEX intraday trading strategies.

Runs all 10 strategy variants (5 base + 5 enhanced) across both indices,
collects performance metrics, ranks them, and identifies the best performing strategy.

Strategies are evaluated on:
1. Win Rate (must exceed 33.3% breakeven for 1:2 RR)
2. Max Drawdown (lower is better)
3. Avg Winning Trade Duration (faster is better)
4. Profit Factor (higher is better)
"""

import sys
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generator import generate_intraday_data
from strategies import (
    MACDCrossoverRSIFilter,
    RSIReversalMomentum,
    TripleConfluence,
    MomentumBreakoutMACD,
    AdaptiveRSIMACD,
    TrendFilteredMACD,
    PullbackMomentum,
    VolatilitySqueezeBreakout,
    EMACrossoverTriple,
    VWAPMomentumMACD,
)
from backtester import BacktestEngine


def create_strategies():
    """Instantiate all strategy variants with optimized parameters."""
    return {
        # ── Base Strategies ──
        "S1: MACD Crossover + RSI Filter": MACDCrossoverRSIFilter(
            rsi_long_min=45, rsi_long_max=65,
            rsi_short_min=35, rsi_short_max=55,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S2: RSI Reversal + Momentum": RSIReversalMomentum(
            rsi_oversold=25, rsi_overbought=70,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S3: Triple Confluence": TripleConfluence(
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S4: Momentum Breakout + MACD": MomentumBreakoutMACD(
            momentum_threshold=0.5,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S5: Adaptive RSI-MACD": AdaptiveRSIMACD(
            lookback=30,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        # ── Enhanced Strategies ──
        "S6: Trend-Filtered MACD": TrendFilteredMACD(
            ema_period=50,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S7: Pullback Momentum": PullbackMomentum(
            ema_period=20, pullback_pct=0.002,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S8: Volatility Squeeze": VolatilitySqueezeBreakout(
            bb_period=20, squeeze_lookback=20,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S9: EMA Crossover Triple": EMACrossoverTriple(
            fast_ema=9, slow_ema=21,
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
        "S10: VWAP Momentum MACD": VWAPMomentumMACD(
            atr_multiplier=1.0, rr_ratio=2.0,
        ),
    }


def run_single_backtest(strategy, data, engine):
    """Run a single strategy backtest and return results."""
    signals_df = strategy.run(data)
    results = engine.run(signals_df)
    return results


def compute_composite_score(metrics):
    """
    Compute composite score for ranking.

    Weights:
    - Win rate: 40% (must be > 33.3% to be profitable at 1:2 RR)
    - Max drawdown: 30% (capital preservation)
    - Avg win duration: 20% (faster trades = less exposure)
    - Profit factor: 10% (overall profitability)
    """
    if metrics["total_trades"] == 0:
        return 0.0

    win_score = metrics["win_rate"]
    dd_score = max(0, 100 - metrics["max_drawdown_pct"] * 10)
    duration_score = max(0, 100 - metrics["avg_win_duration_candles"] * 2)
    pf_score = min(100, metrics["profit_factor"] * 25)

    composite = (
        win_score * 0.40
        + dd_score * 0.30
        + duration_score * 0.20
        + pf_score * 0.10
    )
    return round(composite, 2)


def print_header():
    print("=" * 90)
    print("  NIFTY 50 & SENSEX INTRADAY TRADING STRATEGY BACKTEST")
    print("  Indicators: MACD + RSI + Momentum | Risk-Reward: 1:2")
    print("  Strategies: 5 Base + 5 Enhanced = 10 Total")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print()


def print_strategy_results(name, metrics, symbol):
    """Print detailed results for a single strategy."""
    # Profitability indicator
    profitable = "+" if metrics["win_rate"] >= 33.3 else "-"
    print(f"\n  [{symbol}] [{profitable}] {name}")
    print(f"  {'─' * 65}")
    print(f"  Total Trades:     {metrics['total_trades']:>6}")
    print(f"  Win Rate:         {metrics['win_rate']:>6.1f}%  {'(PROFITABLE at 1:2 RR)' if metrics['win_rate'] >= 33.3 else '(below 33.3% breakeven)'}")
    print(f"  Loss Rate:        {metrics['loss_rate']:>6.1f}%")
    print(f"  Profit Factor:    {metrics['profit_factor']:>6.2f}")
    print(f"  Net P&L:          {metrics['net_pnl']:>10.2f} ({metrics['net_pnl_pct']:+.2f}%)")
    print(f"  Max Drawdown:     {metrics['max_drawdown_pct']:>6.2f}%")
    print(f"  Avg Win Duration: {metrics['avg_win_duration_candles']:>6.1f} candles ({metrics['avg_win_duration_candles'] * 5:.0f} min)")
    print(f"  Avg Loss Duration:{metrics['avg_loss_duration_candles']:>6.1f} candles ({metrics['avg_loss_duration_candles'] * 5:.0f} min)")
    print(f"  Long/Short:       {metrics['long_trades']}/{metrics['short_trades']} "
          f"(Win: {metrics['long_win_rate']:.1f}%/{metrics['short_win_rate']:.1f}%)")
    print(f"  Sharpe Estimate:  {metrics['sharpe_estimate']:>6.2f}")


def main():
    print_header()

    symbols = {
        "NIFTY50": {"seeds": [42, 123, 777, 2024, 9999], "days": 250},
        "SENSEX": {"seeds": [42, 123, 777, 2024, 9999], "days": 250},
    }

    strategies = create_strategies()
    engine = BacktestEngine(
        initial_capital=100000.0,
        position_size_pct=10.0,
        max_trades_per_day=4,
    )

    all_results = []

    for symbol, config in symbols.items():
        print(f"\n{'─' * 90}")
        print(f"  BACKTESTING ON: {symbol} ({config['days']} trading days x {len(config['seeds'])} seeds)")
        print(f"{'─' * 90}")

        for strat_name, strategy in strategies.items():
            seed_metrics = []

            for seed in config["seeds"]:
                data = generate_intraday_data(
                    symbol=symbol, num_days=config["days"], seed=seed
                )
                result = run_single_backtest(strategy, data, engine)
                seed_metrics.append(result["metrics"])

            # Average metrics across seeds
            avg_metrics = {}
            for key in seed_metrics[0]:
                values = [m[key] for m in seed_metrics]
                if isinstance(values[0], (int, float)):
                    avg_metrics[key] = round(np.mean(values), 2)
                else:
                    avg_metrics[key] = values[0]

            avg_metrics["total_trades"] = sum(m["total_trades"] for m in seed_metrics)

            print_strategy_results(strat_name, avg_metrics, symbol)

            score = compute_composite_score(avg_metrics)
            all_results.append({
                "strategy": strat_name,
                "symbol": symbol,
                "metrics": avg_metrics,
                "score": score,
            })

    # ── FULL COMPARISON TABLE ──
    print(f"\n\n{'=' * 90}")
    print("  COMPLETE STRATEGY RANKING (All 10 Strategies x 2 Indices)")
    print(f"{'=' * 90}\n")

    table_data = []
    for rank, r in enumerate(sorted(all_results, key=lambda x: x["score"], reverse=True), 1):
        m = r["metrics"]
        profitable = "Y" if m["win_rate"] >= 33.3 else "N"
        table_data.append([
            rank,
            r["strategy"],
            r["symbol"],
            m["total_trades"],
            f"{m['win_rate']:.1f}%",
            f"{m['max_drawdown_pct']:.2f}%",
            f"{m['avg_win_duration_candles']:.1f}",
            f"{m['profit_factor']:.2f}",
            f"{m['net_pnl_pct']:+.2f}%",
            profitable,
            f"{r['score']:.1f}",
        ])

    headers = [
        "#", "Strategy", "Symbol", "Trades", "Win%", "MaxDD%",
        "AvgDur", "PF", "Net%", "Prof?", "Score"
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # ── TOP 5 ──
    top5 = sorted(all_results, key=lambda x: x["score"], reverse=True)[:5]
    print(f"\n{'=' * 90}")
    print("  TOP 5 STRATEGIES")
    print(f"{'=' * 90}")

    for i, r in enumerate(top5, 1):
        m = r["metrics"]
        print(f"\n  #{i}: {r['strategy']} on {r['symbol']}")
        print(f"      Score: {r['score']} | Win Rate: {m['win_rate']:.1f}% | "
              f"MaxDD: {m['max_drawdown_pct']:.2f}% | "
              f"AvgDur: {m['avg_win_duration_candles']:.1f} candles ({m['avg_win_duration_candles']*5:.0f} min) | "
              f"PF: {m['profit_factor']:.2f} | Net: {m['net_pnl_pct']:+.2f}%")

    # ── BEST OVERALL ──
    best = top5[0]
    print(f"\n{'=' * 90}")
    print(f"  BEST PERFORMING STRATEGY")
    print(f"{'=' * 90}")
    print(f"\n  >>> {best['strategy']} on {best['symbol']} <<<")
    print(f"  Composite Score: {best['score']}")
    print(f"\n  Detailed Metrics:")
    print(f"    Win Rate:             {best['metrics']['win_rate']:.1f}%")
    print(f"    Max Drawdown:         {best['metrics']['max_drawdown_pct']:.2f}%")
    print(f"    Avg Win Duration:     {best['metrics']['avg_win_duration_candles']:.1f} candles "
          f"({best['metrics']['avg_win_duration_candles'] * 5:.0f} minutes)")
    print(f"    Avg Loss Duration:    {best['metrics']['avg_loss_duration_candles']:.1f} candles "
          f"({best['metrics']['avg_loss_duration_candles'] * 5:.0f} minutes)")
    print(f"    Profit Factor:        {best['metrics']['profit_factor']:.2f}")
    print(f"    Net Return:           {best['metrics']['net_pnl_pct']:+.2f}%")
    print(f"    Sharpe Estimate:      {best['metrics']['sharpe_estimate']:.2f}")
    print(f"    Long Win Rate:        {best['metrics']['long_win_rate']:.1f}%")
    print(f"    Short Win Rate:       {best['metrics']['short_win_rate']:.1f}%")
    print(f"    Total Trades (5 runs):{best['metrics']['total_trades']}")

    # ── STRATEGY CATEGORY INSIGHTS ──
    print(f"\n{'=' * 90}")
    print(f"  STRATEGY INSIGHTS BY CATEGORY")
    print(f"{'=' * 90}")

    strat_scores = {}
    for r in all_results:
        name = r["strategy"]
        if name not in strat_scores:
            strat_scores[name] = []
        strat_scores[name].append(r)

    base_strats = {k: v for k, v in strat_scores.items() if k.startswith("S") and int(k[1]) <= 5}
    enhanced_strats = {k: v for k, v in strat_scores.items() if k.startswith("S") and k[1:3].strip(":").isdigit() and int(k[1:3].strip(":")) > 5}

    # Handle the S10 case properly
    enhanced_strats = {}
    base_strats = {}
    for k, v in strat_scores.items():
        # Extract number from "S1:", "S10:" etc.
        num_str = k.split(":")[0][1:]
        num = int(num_str)
        if num <= 5:
            base_strats[k] = v
        else:
            enhanced_strats[k] = v

    print("\n  --- Base Strategies (S1-S5) ---")
    for name, results in sorted(base_strats.items(), key=lambda x: np.mean([r["score"] for r in x[1]]), reverse=True):
        avg_score = np.mean([r["score"] for r in results])
        avg_wr = np.mean([r["metrics"]["win_rate"] for r in results])
        avg_dd = np.mean([r["metrics"]["max_drawdown_pct"] for r in results])
        avg_pf = np.mean([r["metrics"]["profit_factor"] for r in results])
        print(f"  {name}: Score={avg_score:.1f} WR={avg_wr:.1f}% DD={avg_dd:.2f}% PF={avg_pf:.2f}")

    print("\n  --- Enhanced Strategies (S6-S10) ---")
    for name, results in sorted(enhanced_strats.items(), key=lambda x: np.mean([r["score"] for r in x[1]]), reverse=True):
        avg_score = np.mean([r["score"] for r in results])
        avg_wr = np.mean([r["metrics"]["win_rate"] for r in results])
        avg_dd = np.mean([r["metrics"]["max_drawdown_pct"] for r in results])
        avg_pf = np.mean([r["metrics"]["profit_factor"] for r in results])
        print(f"  {name}: Score={avg_score:.1f} WR={avg_wr:.1f}% DD={avg_dd:.2f}% PF={avg_pf:.2f}")

    # ── KEY FINDINGS ──
    print(f"\n{'=' * 90}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 90}")

    profitable_strats = [r for r in all_results if r["metrics"]["win_rate"] >= 33.3]
    if profitable_strats:
        print(f"\n  Strategies exceeding 33.3% breakeven win rate (profitable at 1:2 RR):")
        for r in sorted(profitable_strats, key=lambda x: x["metrics"]["win_rate"], reverse=True):
            m = r["metrics"]
            print(f"    - {r['strategy']} [{r['symbol']}]: WR={m['win_rate']:.1f}% PF={m['profit_factor']:.2f}")
    else:
        print(f"\n  No strategy exceeded the 33.3% breakeven win rate for 1:2 RR.")
        print(f"  Closest strategies:")
        closest = sorted(all_results, key=lambda x: x["metrics"]["win_rate"], reverse=True)[:3]
        for r in closest:
            m = r["metrics"]
            print(f"    - {r['strategy']} [{r['symbol']}]: WR={m['win_rate']:.1f}% (need +{33.3-m['win_rate']:.1f}pp)")

    print(f"\n  Lowest Max Drawdown:")
    lowest_dd = sorted(all_results, key=lambda x: x["metrics"]["max_drawdown_pct"])[:3]
    for r in lowest_dd:
        print(f"    - {r['strategy']} [{r['symbol']}]: {r['metrics']['max_drawdown_pct']:.2f}%")

    print(f"\n  Fastest Winning Trades:")
    fastest = sorted(all_results, key=lambda x: x["metrics"]["avg_win_duration_candles"] if x["metrics"]["avg_win_duration_candles"] > 0 else 999)[:3]
    for r in fastest:
        dur = r["metrics"]["avg_win_duration_candles"]
        print(f"    - {r['strategy']} [{r['symbol']}]: {dur:.1f} candles ({dur*5:.0f} min)")

    print(f"\n{'=' * 90}")
    print("  RISK DISCLAIMER: This is for educational/research purposes only.")
    print("  Past (simulated) performance does not guarantee future results.")
    print("  Always paper-trade before using real capital.")
    print(f"{'=' * 90}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    serializable_results = []
    for r in all_results:
        serializable_results.append({
            "strategy": r["strategy"],
            "symbol": r["symbol"],
            "score": r["score"],
            "metrics": r["metrics"],
        })

    with open(output_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "config": {
                "initial_capital": 100000,
                "position_size_pct": 10,
                "rr_ratio": "1:2",
                "atr_multiplier": 1.0,
                "max_trades_per_day": 4,
                "data_interval": "5min",
                "trading_days_per_seed": 250,
                "seeds_per_symbol": 5,
            },
            "results": serializable_results,
            "best_strategy": {
                "name": best["strategy"],
                "symbol": best["symbol"],
                "score": best["score"],
                "metrics": best["metrics"],
            },
        }, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    main()
