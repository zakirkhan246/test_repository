#!/usr/bin/env python3
"""
Parameter optimizer for intraday strategies.

Sweeps ATR multipliers, RSI thresholds, and momentum thresholds
to find parameter combinations that push win rates above the
33.3% breakeven threshold for 1:2 RR strategies.
"""

import sys
import os
import json
from itertools import product

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generator import generate_intraday_data
from strategies import (
    MACDCrossoverRSIFilter,
    RSIReversalMomentum,
    TripleConfluence,
    MomentumBreakoutMACD,
    AdaptiveRSIMACD,
)
from backtester import BacktestEngine


def optimize_strategy_1():
    """Optimize MACD Crossover + RSI Filter."""
    print("\n  Optimizing S1: MACD Crossover + RSI Filter...")
    best = {"score": 0}

    atr_mults = [1.0, 1.2, 1.5, 2.0]
    rsi_long_mins = [35, 40, 45]
    rsi_long_maxs = [65, 70, 75]

    engine = BacktestEngine(initial_capital=100000, position_size_pct=10, max_trades_per_day=4)

    for atr_m, rl_min, rl_max in product(atr_mults, rsi_long_mins, rsi_long_maxs):
        if rl_min >= rl_max:
            continue

        strategy = MACDCrossoverRSIFilter(
            rsi_long_min=rl_min, rsi_long_max=rl_max,
            rsi_short_min=100 - rl_max, rsi_short_max=100 - rl_min,
            atr_multiplier=atr_m, rr_ratio=2.0,
        )

        metrics_list = []
        for symbol in ["NIFTY50", "SENSEX"]:
            for seed in [42, 123, 777]:
                data = generate_intraday_data(symbol=symbol, num_days=250, seed=seed)
                signals_df = strategy.run(data)
                result = engine.run(signals_df)
                metrics_list.append(result["metrics"])

        avg_wr = np.mean([m["win_rate"] for m in metrics_list])
        avg_dd = np.mean([m["max_drawdown_pct"] for m in metrics_list])
        avg_dur = np.mean([m["avg_win_duration_candles"] for m in metrics_list])
        avg_pf = np.mean([m["profit_factor"] for m in metrics_list])
        avg_trades = np.mean([m["total_trades"] for m in metrics_list])

        # Score: prioritize win rate, then low DD, then fast trades
        score = avg_wr * 0.5 + max(0, 100 - avg_dd * 10) * 0.25 + max(0, 100 - avg_dur * 2) * 0.15 + min(100, avg_pf * 25) * 0.10

        if score > best["score"] and avg_trades >= 10:
            best = {
                "score": score, "params": {"atr_m": atr_m, "rsi_long_min": rl_min, "rsi_long_max": rl_max},
                "win_rate": avg_wr, "max_dd": avg_dd, "avg_dur": avg_dur, "pf": avg_pf, "trades": avg_trades,
            }

    print(f"    Best params: ATR={best['params']['atr_m']}, RSI=[{best['params']['rsi_long_min']},{best['params']['rsi_long_max']}]")
    print(f"    Win Rate: {best['win_rate']:.1f}% | MaxDD: {best['max_dd']:.2f}% | AvgDur: {best['avg_dur']:.1f} | PF: {best['pf']:.2f}")
    return best


def optimize_strategy_2():
    """Optimize RSI Reversal + Momentum."""
    print("\n  Optimizing S2: RSI Reversal + Momentum...")
    best = {"score": 0}

    atr_mults = [1.0, 1.2, 1.5, 2.0]
    rsi_oversolds = [25, 30, 35, 40]
    rsi_overboughts = [60, 65, 70, 75]

    engine = BacktestEngine(initial_capital=100000, position_size_pct=10, max_trades_per_day=4)

    for atr_m, rsi_os, rsi_ob in product(atr_mults, rsi_oversolds, rsi_overboughts):
        strategy = RSIReversalMomentum(
            rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
            atr_multiplier=atr_m, rr_ratio=2.0,
        )

        metrics_list = []
        for symbol in ["NIFTY50", "SENSEX"]:
            for seed in [42, 123, 777]:
                data = generate_intraday_data(symbol=symbol, num_days=250, seed=seed)
                signals_df = strategy.run(data)
                result = engine.run(signals_df)
                metrics_list.append(result["metrics"])

        avg_wr = np.mean([m["win_rate"] for m in metrics_list])
        avg_dd = np.mean([m["max_drawdown_pct"] for m in metrics_list])
        avg_dur = np.mean([m["avg_win_duration_candles"] for m in metrics_list])
        avg_pf = np.mean([m["profit_factor"] for m in metrics_list])
        avg_trades = np.mean([m["total_trades"] for m in metrics_list])

        score = avg_wr * 0.5 + max(0, 100 - avg_dd * 10) * 0.25 + max(0, 100 - avg_dur * 2) * 0.15 + min(100, avg_pf * 25) * 0.10

        if score > best["score"] and avg_trades >= 10:
            best = {
                "score": score, "params": {"atr_m": atr_m, "rsi_oversold": rsi_os, "rsi_overbought": rsi_ob},
                "win_rate": avg_wr, "max_dd": avg_dd, "avg_dur": avg_dur, "pf": avg_pf, "trades": avg_trades,
            }

    print(f"    Best params: ATR={best['params']['atr_m']}, RSI_OS={best['params']['rsi_oversold']}, RSI_OB={best['params']['rsi_overbought']}")
    print(f"    Win Rate: {best['win_rate']:.1f}% | MaxDD: {best['max_dd']:.2f}% | AvgDur: {best['avg_dur']:.1f} | PF: {best['pf']:.2f}")
    return best


def optimize_strategy_3():
    """Optimize Triple Confluence."""
    print("\n  Optimizing S3: Triple Confluence...")
    best = {"score": 0}

    atr_mults = [1.0, 1.2, 1.5, 2.0]
    engine = BacktestEngine(initial_capital=100000, position_size_pct=10, max_trades_per_day=4)

    for atr_m in atr_mults:
        strategy = TripleConfluence(atr_multiplier=atr_m, rr_ratio=2.0)

        metrics_list = []
        for symbol in ["NIFTY50", "SENSEX"]:
            for seed in [42, 123, 777]:
                data = generate_intraday_data(symbol=symbol, num_days=250, seed=seed)
                signals_df = strategy.run(data)
                result = engine.run(signals_df)
                metrics_list.append(result["metrics"])

        avg_wr = np.mean([m["win_rate"] for m in metrics_list])
        avg_dd = np.mean([m["max_drawdown_pct"] for m in metrics_list])
        avg_dur = np.mean([m["avg_win_duration_candles"] for m in metrics_list])
        avg_pf = np.mean([m["profit_factor"] for m in metrics_list])
        avg_trades = np.mean([m["total_trades"] for m in metrics_list])

        score = avg_wr * 0.5 + max(0, 100 - avg_dd * 10) * 0.25 + max(0, 100 - avg_dur * 2) * 0.15 + min(100, avg_pf * 25) * 0.10

        if score > best["score"] and avg_trades >= 10:
            best = {
                "score": score, "params": {"atr_m": atr_m},
                "win_rate": avg_wr, "max_dd": avg_dd, "avg_dur": avg_dur, "pf": avg_pf, "trades": avg_trades,
            }

    print(f"    Best params: ATR={best['params']['atr_m']}")
    print(f"    Win Rate: {best['win_rate']:.1f}% | MaxDD: {best['max_dd']:.2f}% | AvgDur: {best['avg_dur']:.1f} | PF: {best['pf']:.2f}")
    return best


def optimize_strategy_4():
    """Optimize Momentum Breakout + MACD."""
    print("\n  Optimizing S4: Momentum Breakout + MACD...")
    best = {"score": 0}

    atr_mults = [1.0, 1.2, 1.5, 2.0]
    mom_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    engine = BacktestEngine(initial_capital=100000, position_size_pct=10, max_trades_per_day=4)

    for atr_m, mom_t in product(atr_mults, mom_thresholds):
        strategy = MomentumBreakoutMACD(
            momentum_threshold=mom_t,
            atr_multiplier=atr_m, rr_ratio=2.0,
        )

        metrics_list = []
        for symbol in ["NIFTY50", "SENSEX"]:
            for seed in [42, 123, 777]:
                data = generate_intraday_data(symbol=symbol, num_days=250, seed=seed)
                signals_df = strategy.run(data)
                result = engine.run(signals_df)
                metrics_list.append(result["metrics"])

        avg_wr = np.mean([m["win_rate"] for m in metrics_list])
        avg_dd = np.mean([m["max_drawdown_pct"] for m in metrics_list])
        avg_dur = np.mean([m["avg_win_duration_candles"] for m in metrics_list])
        avg_pf = np.mean([m["profit_factor"] for m in metrics_list])
        avg_trades = np.mean([m["total_trades"] for m in metrics_list])

        score = avg_wr * 0.5 + max(0, 100 - avg_dd * 10) * 0.25 + max(0, 100 - avg_dur * 2) * 0.15 + min(100, avg_pf * 25) * 0.10

        if score > best["score"] and avg_trades >= 10:
            best = {
                "score": score, "params": {"atr_m": atr_m, "momentum_threshold": mom_t},
                "win_rate": avg_wr, "max_dd": avg_dd, "avg_dur": avg_dur, "pf": avg_pf, "trades": avg_trades,
            }

    print(f"    Best params: ATR={best['params']['atr_m']}, Mom Threshold={best['params']['momentum_threshold']}")
    print(f"    Win Rate: {best['win_rate']:.1f}% | MaxDD: {best['max_dd']:.2f}% | AvgDur: {best['avg_dur']:.1f} | PF: {best['pf']:.2f}")
    return best


def optimize_strategy_5():
    """Optimize Adaptive RSI-MACD."""
    print("\n  Optimizing S5: Adaptive RSI-MACD...")
    best = {"score": 0}

    atr_mults = [1.0, 1.2, 1.5, 2.0]
    lookbacks = [30, 50, 75]

    engine = BacktestEngine(initial_capital=100000, position_size_pct=10, max_trades_per_day=4)

    for atr_m, lb in product(atr_mults, lookbacks):
        strategy = AdaptiveRSIMACD(
            lookback=lb,
            atr_multiplier=atr_m, rr_ratio=2.0,
        )

        metrics_list = []
        for symbol in ["NIFTY50", "SENSEX"]:
            for seed in [42, 123, 777]:
                data = generate_intraday_data(symbol=symbol, num_days=250, seed=seed)
                signals_df = strategy.run(data)
                result = engine.run(signals_df)
                metrics_list.append(result["metrics"])

        avg_wr = np.mean([m["win_rate"] for m in metrics_list])
        avg_dd = np.mean([m["max_drawdown_pct"] for m in metrics_list])
        avg_dur = np.mean([m["avg_win_duration_candles"] for m in metrics_list])
        avg_pf = np.mean([m["profit_factor"] for m in metrics_list])
        avg_trades = np.mean([m["total_trades"] for m in metrics_list])

        score = avg_wr * 0.5 + max(0, 100 - avg_dd * 10) * 0.25 + max(0, 100 - avg_dur * 2) * 0.15 + min(100, avg_pf * 25) * 0.10

        if score > best["score"] and avg_trades >= 10:
            best = {
                "score": score, "params": {"atr_m": atr_m, "lookback": lb},
                "win_rate": avg_wr, "max_dd": avg_dd, "avg_dur": avg_dur, "pf": avg_pf, "trades": avg_trades,
            }

    print(f"    Best params: ATR={best['params']['atr_m']}, Lookback={best['params']['lookback']}")
    print(f"    Win Rate: {best['win_rate']:.1f}% | MaxDD: {best['max_dd']:.2f}% | AvgDur: {best['avg_dur']:.1f} | PF: {best['pf']:.2f}")
    return best


def main():
    print("=" * 80)
    print("  PARAMETER OPTIMIZATION")
    print("  Finding best parameters for each strategy variant")
    print("=" * 80)

    results = {}
    results["S1"] = optimize_strategy_1()
    results["S2"] = optimize_strategy_2()
    results["S3"] = optimize_strategy_3()
    results["S4"] = optimize_strategy_4()
    results["S5"] = optimize_strategy_5()

    print(f"\n\n{'=' * 80}")
    print("  OPTIMIZATION RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    table = []
    for name, r in sorted(results.items(), key=lambda x: x[1]["score"], reverse=True):
        table.append([
            name, f"{r['win_rate']:.1f}%", f"{r['max_dd']:.2f}%",
            f"{r['avg_dur']:.1f}", f"{r['pf']:.2f}", f"{r['trades']:.0f}",
            f"{r['score']:.1f}", str(r["params"]),
        ])

    print(tabulate(table, headers=["Strategy", "WinRate", "MaxDD", "AvgDur", "PF", "Trades", "Score", "Params"], tablefmt="grid"))

    # Save optimized params
    with open(os.path.join(os.path.dirname(__file__), "optimized_params.json"), "w") as f:
        serializable = {k: {**v, "params": v["params"]} for k, v in results.items()}
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Optimized parameters saved to optimized_params.json")

    best_name = max(results.items(), key=lambda x: x[1]["score"])
    print(f"\n  OVERALL BEST: {best_name[0]} with score {best_name[1]['score']:.1f}")
    print(f"  Win Rate: {best_name[1]['win_rate']:.1f}% | Params: {best_name[1]['params']}")

    return results


if __name__ == "__main__":
    main()
