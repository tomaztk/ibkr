# Backetesting and Trading Dashboard

A comprehensive Python trading dashboard built with Streamlit for historical data analysis, strategy backtesting, and visualization.

## Features

- **Historical Data Fetching**: Download 6 years of minute-level data from Alpaca Markets
- **Interactive Charts**: Candlestick charts with volume, technical indicators, and buy/sell signals
- **Multiple Trading Strategies**:
  - SMA Crossover
  - RSI (Relative Strength Index)
  - Momentum
  - Bollinger Bands
  - MACD
- **Realistic Backtesting**: Includes commission, slippage, and position sizing
- **Comprehensive Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, and more
- **Trade History**: Detailed trade log with entry/exit points and P&L

## Supported Symbols

- **SPY**: S&P 500 ETF
- **QQQ**: Nasdaq 100 ETF
- **DIA**: Dow Jones ETF
- **VTI**: Total Stock Market ETF
- **IWM**: Russell 2000 ETF
- **GLD**: Gold ETF
- **TLT**: 20+ Year Treasury Bond ETF

## Installation

### 1. Clone or Download the Project

```bash
cd trading_dashboard
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Credentials

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Alpaca API credentials:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Getting Alpaca API Keys:**
1. Go to [https://app.alpaca.markets](https://app.alpaca.markets)
2. Sign up for a free account
3. Navigate to Paper Trading → API Keys
4. Generate new API keys

### 5. Download Historical Data

```bash
   python scripts/download_data.py                    # Default
   python scripts/download_data.py --timeframe 1Day  # Daily data
   python scripts/download_data.py --force           # Re-download
```

This will download minute-level data from January 1, 2020 to today for all configured symbols. The data is stored in parquet format in the `data/` directory.

**Note:** This may take some time depending on your internet connection (~15-30 minutes for all symbols).

### 6. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Project Structure

```
trading_dashboard/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Your API credentials (create this)
├── README.md                # This file
│
├── data/                    # Parquet data files (created after download)
│   ├── SPY_1Min.parquet
│   ├── QQQ_1Min.parquet
│   └── ...
│
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── data_fetcher.py      # Alpaca data fetcher
│
├── strategies/              # Trading strategies
│   ├── __init__.py
│   └── strategies.py        # Strategy implementations
│
├── backtest/                # Backtesting engine
│   ├── __init__.py
│   └── engine.py            # Backtest engine
│
└── scripts/                 # Utility scripts
    ├── __init__.py
    └── download_data.py     # Data download script
```

## Usage Guide

### Dashboard Navigation

1. **Sidebar Settings**:
   - Select symbol and timeframe
   - Choose date range for analysis
   - Select trading strategy and adjust parameters
   - Configure backtest settings (capital, commission, slippage)

2. **Price Chart Tab**:
   - Interactive candlestick chart
   - Toggle volume display
   - Add technical indicators
   - View buy/sell signals

3. **Backtest Results Tab**:
   - Key performance metrics
   - Equity curve visualization
   - Drawdown analysis
   - Trade statistics

4. **Trade History Tab**:
   - Detailed trade log
   - Entry/exit times and prices
   - P&L for each trade

5. **Analysis Tab**:
   - Price statistics
   - Returns distribution
   - Volume analysis

### Strategy Parameters

#### SMA Crossover
- `fast_period`: Fast moving average period (default: 10)
- `slow_period`: Slow moving average period (default: 50)

#### RSI
- `period`: RSI calculation period (default: 14)
- `overbought`: Overbought threshold (default: 70)
- `oversold`: Oversold threshold (default: 30)

#### Momentum
- `period`: Lookback period for momentum (default: 20)
- `threshold`: Entry threshold (default: 0)

#### Bollinger Bands
- `period`: Moving average period (default: 20)
- `std_dev`: Standard deviation multiplier (default: 2.0)

### Backtest Settings

- **Initial Capital**: Starting portfolio value
- **Commission**: Transaction cost per trade (percentage)
- **Slippage**: Price slippage estimation (percentage)
- **Position Size**: Percentage of capital to use per trade

## Customization

### Adding New Symbols

Edit `config.py`:

```python
SYMBOLS = ["SPY", "QQQ", "DIA", "VTI", "IWM", "GLD", "TLT", "AAPL"]  # Add your symbols
```

### Adding New Strategies

1. Create a new strategy class in `strategies/strategies.py`:

```python
class MyStrategy(BaseStrategy):
    def get_default_params(self):
        return {'param1': 10}
    
    def generate_signals(self, df):
        df = df.copy()
        # Your strategy logic here
        df['signal'] = 0
        # Set df['signal'] = 1 for long, -1 for short
        df['buy_signal'] = ...
        df['sell_signal'] = ...
        return df
```

2. Register in STRATEGIES dict:
```python
STRATEGIES = {
    # ... existing strategies
    'My Strategy': MyStrategy
}
```

### Changing Date Range

Edit `config.py`:

```python
START_DATE = "2018-01-01"  # Change start date
END_DATE = datetime.now().strftime("%Y-%m-%d")
```

Then re-download data:
```bash
python scripts/download_data.py
```

## Troubleshooting

### "Missing Alpaca credentials"
- Ensure `.env` file exists and contains valid API keys
- Check that keys are not enclosed in quotes

### "No data available"
- Run `python scripts/download_data.py` first
- Check that parquet files exist in `data/` directory

### Rate Limiting
- Alpaca has rate limits; the fetcher includes delays to avoid this
- If you see errors, wait a few minutes and try again

### Memory Issues with Large Data
- Consider using a smaller date range
- Use higher timeframes (15Min, 1Hour, 1Day) for initial testing

## Dependencies

- **alpaca-trade-api**: Alpaca Markets API client
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **streamlit**: Web dashboard framework
- **plotly**: Interactive charts
- **ta**: Technical analysis indicators
- **pyarrow**: Parquet file support
- **python-dotenv**: Environment variable management

