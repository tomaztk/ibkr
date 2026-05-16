#!/usr/bin/env python3

"""
Fetch historical OHLCV data for AAPL and MSFT from Alpaca Markets.
Saves daily and hourly bars to Parquet files.

env:
 python3 -m venv ~/envs/myenv_trading
 source ~/envs/myenv_trading/bin/activate
 deactivate

 VSCode:
 1. Open VS Code, press Cmd+Shift+P → type "Python: Select Interpreter"
 2. Click "Enter interpreter path" → paste:    - env: ~/envs/myenv_trading/bin/python3

 version: Pyhthon 3.13.9 (homebrew)

 pip:

- pip list

Requirements:
    python3.9 -m pip install --upgrade pip
    pip3 install alpaca-py pandas pyarrow python-dotenv
    python3 01_access_Data.py   
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

SYMBOLS    = ["AAPL", "MSFT"]
OUTPUT_DIR = Path("alpaca_data")
OUTPUT_DIR.mkdir(exist_ok=True)

END_DATE   = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=5 * 3)   # ~5 years

RESOLUTIONS = {
    "daily":  TimeFrame(1, TimeFrameUnit.Day),
    "hourly": TimeFrame(1, TimeFrameUnit.Hour),
}

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
client = StockHistoricalDataClient(API_KEY, API_SECRET)


# ---------------------------------------------------------------------------
# Fetch & save
# ---------------------------------------------------------------------------
def fetch_and_save(symbol: str, label: str, timeframe: TimeFrame) -> None:
    print(f"  Fetching {symbol} [{label}] from {START_DATE.date()} to {END_DATE.date()} …")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=START_DATE,
        end=END_DATE,
        feed="iex",           # free data feed; change to "sip" for paid plan
        adjustment="all",     # split- and dividend-adjusted
    )

    bars = client.get_stock_bars(request)
    df: pd.DataFrame = bars.df

    if df.empty:
        print(f"    WARNING: no data returned for {symbol} [{label}]")
        return

    # Flatten multi-index if present (symbol, timestamp) → reset index
    df = df.reset_index()

    # Ensure timestamp column is timezone-aware UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    out_path = OUTPUT_DIR / f"{symbol}_{label}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    print(f"    ✓ Saved {len(df):,} rows → {out_path}")
    print(f"      Columns : {list(df.columns)}")
    print(f"      Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")


def main() -> None:
    print("=" * 60)
    print("Alpaca Historical Data Downloader")
    print("=" * 60)

    for symbol in SYMBOLS:
        print(f"\n{'─' * 40}")
        print(f" Symbol: {symbol}")
        print(f"{'─' * 40}")
        for label, timeframe in RESOLUTIONS.items():
            fetch_and_save(symbol, label, timeframe)

    print("\n✅ All done. Files written to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()