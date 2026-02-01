"""
Configuration settings for the trading dashboard.
"""
from datetime import datetime
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Symbols to track
SYMBOLS = ["SPY", "QQQ", "DIA", "VTI", "IWM", "GLD", "TLT"]

# Symbol descriptions
SYMBOL_DESCRIPTIONS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "DIA": "Dow Jones ETF",
    "VTI": "Total Stock Market ETF",
    "IWM": "Russell 2000 ETF",
    "GLD": "Gold ETF",
    "TLT": "20+ Year Treasury Bond ETF"
}

# Date range for historical data
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Timeframes available
TIMEFRAMES = {
    "1Min": "1Min",
    "5Min": "5Min",
    "15Min": "15Min",
    "1Hour": "1Hour",
    "1Day": "1Day"
}

# Default timeframe for download
DEFAULT_TIMEFRAME = "1Min"

# Backtest settings
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_COMMISSION = 0.001  # 0.1% per trade
DEFAULT_SLIPPAGE = 0.0005   # 0.05% slippage

# Strategy parameters defaults
STRATEGY_PARAMS = {
    "SMA_Crossover": {
        "fast_period": 10,
        "slow_period": 50
    },
    "RSI": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    "Momentum": {
        "period": 20,
        "threshold": 0.0
    },
    "Bollinger_Bands": {
        "period": 20,
        "std_dev": 2.0
    }
}
