#!/usr/bin/env python3
"""
Script to download all historical data from Alpaca.
Run this before using the dashboard.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SYMBOLS, START_DATE, END_DATE, DEFAULT_TIMEFRAME, DATA_DIR
from utils.data_fetcher import AlpacaDataFetcher


def main():
    """Download all historical data."""
    print("=" * 60)
    print("Trading Dashboard - Data Downloader")
    print("=" * 60)
    print()
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Timeframe: {DEFAULT_TIMEFRAME}")
    print(f"Data Directory: {DATA_DIR}")
    print()
    
    try:
        fetcher = AlpacaDataFetcher(data_dir=str(DATA_DIR))
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease create a .env file with your Alpaca credentials.")
        print("Copy .env.example to .env and add your API keys.")
        return 1
    
    print("Starting download...")
    print("-" * 60)
    
    results = fetcher.fetch_multiple_symbols(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=DEFAULT_TIMEFRAME,
        force_refresh=False
    )
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    total_rows = 0
    for symbol, df in results.items():
        if not df.empty:
            rows = len(df)
            total_rows += rows
            print(f"OK - {symbol}: {rows:,} rows ({df.index.min().date()} to {df.index.max().date()})")
        else:
            print(f"NOK - {symbol}: No data")
    
    print("-" * 60)
    print(f"Total: {total_rows:,} rows downloaded")
    
    # Show file sizes
    print("\nCached Files:")
    for name, info in fetcher.get_available_data().items():
        print(f"  {name}: {info['rows']:,} rows, {info['size_mb']:.2f} MB")
    
    print("\nOK - Data download complete! You can now run the dashboard.")    
    return 0

if __name__ == "__main__":
    sys.exit(main())
