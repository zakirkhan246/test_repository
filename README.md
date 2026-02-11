# NIFTY 50 & SENSEX Intraday Trading Strategy

## Overview
Intraday trading strategies for NIFTY 50 and SENSEX using **MACD**, **RSI**, and **Momentum** indicators with a strict **1:2 risk-reward ratio**.

## Strategy Variants

| # | Strategy | Entry Logic |
|---|----------|-------------|
| 1 | **MACD Crossover + RSI Filter** | MACD bullish/bearish crossover confirmed by RSI zones |
| 2 | **RSI Reversal + Momentum Confirmation** | RSI oversold/overbought with momentum divergence |
| 3 | **Triple Confluence** | All three indicators (MACD + RSI + Momentum) must agree |
| 4 | **Momentum Breakout + MACD Trend** | Strong momentum surge with MACD trend confirmation |
| 5 | **Adaptive RSI-MACD** | Dynamic RSI thresholds based on MACD histogram strength |

## Performance Metrics
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Avg Trade Duration**: Mean time to complete winning trades
- **Profit Factor**: Gross profits / Gross losses
- **Risk-Reward Ratio**: Fixed at 1:2

## Project Structure
```
├── data/
│   └── generator.py          # Realistic intraday data generation
├── strategies/
│   ├── __init__.py
│   ├── indicators.py         # MACD, RSI, Momentum calculations
│   ├── base_strategy.py      # Base strategy class
│   └── strategy_variants.py  # All 5 strategy implementations
├── backtester/
│   ├── __init__.py
│   ├── engine.py             # Backtesting engine
│   └── metrics.py            # Performance metrics calculation
├── run_backtest.py           # Main entry point
├── requirements.txt
└── README.md
```

## Usage
```bash
python run_backtest.py
```

## Risk Disclaimer
This is for **educational and research purposes only**. Past performance does not guarantee future results. Always paper-trade before using real capital.
