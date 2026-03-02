#!/usr/bin/env python3
"""
Backtest runner for pure MACD crossover strategy on real SENSEX data.
"""

import sys
import os
import json
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_sensex_data
from strategies import MACDCrossover
from backtester import BacktestEngine


def main():
    print("=" * 70)
    print("  SENSEX INTRADAY BACKTEST — Pure MACD Crossover")
    print("  Entry: MACD/Signal crossover + histogram acceleration")
    print("  Risk: ATR stop loss | Reward: 1:2 RR")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load real data
    data = load_sensex_data()
    n_days = data["date"].nunique()
    print(f"\n  Data: {n_days} trading days of real SENSEX 5-min candles")
    print(f"  Range: {data['date'].min()} to {data['date'].max()}")

    strategy = MACDCrossover(atr_multiplier=1.0, rr_ratio=2.0)
    engine = BacktestEngine(
        initial_capital=100000.0,
        position_size_pct=10.0,
        max_trades_per_day=4,
    )

    # Run
    signals_df = strategy.run(data)
    result = engine.run(signals_df)
    m = result["metrics"]

    # Results
    profitable = "YES" if m["win_rate"] >= 33.3 else "NO"

    print(f"\n{'─' * 70}")
    print(f"  RESULTS")
    print(f"{'─' * 70}")
    print(f"  Total Trades:      {m['total_trades']}")
    print(f"  Win Rate:          {m['win_rate']:.1f}%  (breakeven at 33.3% for 1:2 RR)")
    print(f"  Profitable:        {profitable}")
    print(f"  Loss Rate:         {m['loss_rate']:.1f}%")
    print(f"  Profit Factor:     {m['profit_factor']:.2f}")
    print(f"  Net P&L:           {m['net_pnl']:+.2f} ({m['net_pnl_pct']:+.2f}%)")
    print(f"  Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Estimate:   {m['sharpe_estimate']:.2f}")
    print(f"\n  Avg Win Duration:  {m['avg_win_duration_candles']:.1f} candles ({m['avg_win_duration_candles'] * 5:.0f} min)")
    print(f"  Avg Loss Duration: {m['avg_loss_duration_candles']:.1f} candles ({m['avg_loss_duration_candles'] * 5:.0f} min)")
    print(f"\n  Long Trades:       {m['long_trades']}  (Win: {m['long_win_rate']:.1f}%)")
    print(f"  Short Trades:      {m['short_trades']}  (Win: {m['short_win_rate']:.1f}%)")
    print(f"  Max Consec Wins:   {m['max_consecutive_wins']}")
    print(f"  Max Consec Losses: {m['max_consecutive_losses']}")

    print(f"\n{'─' * 70}")
    print("  RISK DISCLAIMER: Educational/research only.")
    print("  Past performance does not guarantee future results.")
    print(f"{'─' * 70}\n")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "run_date": datetime.now().isoformat(),
            "strategy": "MACD Crossover",
            "config": {
                "initial_capital": 100000,
                "position_size_pct": 10,
                "rr_ratio": "1:2",
                "atr_multiplier": 1.0,
                "max_trades_per_day": 4,
                "data_interval": "5min",
                "data_source": "real SENSEX 1-min resampled to 5-min",
                "trading_days": n_days,
            },
            "metrics": m,
        }, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}")
    return result


if __name__ == "__main__":
    main()
