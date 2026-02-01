"""
Data fetcher module for downloading historical data from Alpaca.
Uses direct REST API calls for reliable data fetching.
Handles pagination and stores data in parquet format.
"""
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
import time
from dotenv import load_dotenv

load_dotenv()

class AlpacaDataFetcher:
    """
    Fetches historical stock data from Alpaca REST API and stores in parquet format.
    """
    
    BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data fetcher with Alpaca credentials.
        
        Args:
            data_dir: Directory to store parquet files
        """
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_API_SECRET")
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Missing Alpaca credentials. Please set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in your .env file."
            )
        
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get the parquet file path for a symbol and timeframe."""
        return self.data_dir / f"{symbol}_{timeframe}.parquet"
    
    def fetch_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol using pagination.
        
        Args:
            symbol: Stock symbol (e.g., "SPY")
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            timeframe: Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            show_progress: Show progress output
            
        Returns:
            DataFrame with all historical data
        """
        # Format dates for API (ISO 8601 format)
        start_dt = pd.Timestamp(start_date, tz='UTC')
        end_dt = pd.Timestamp(end_date, tz='UTC')
        
        start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,  # Max allowed per request
            "adjustment": "raw", # will change the adjustment!!!! must be app! or smth!
            "feed": "sip",
            "sort": "asc",
        }
        
        all_dfs = []
        page = 0
        total_rows = 0
        
        while True:
            try:
                response = requests.get(
                    self.BASE_URL, 
                    headers=self.headers, 
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                payload = response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {symbol}: {e}")
                if page > 0:
                    break
                return pd.DataFrame()
            

            bars = payload.get("bars", {}).get(symbol, [])
            if not bars:
                break
            
         
            df = pd.DataFrame(bars).rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trade_count",
                "vw": "vwap",
            })
            
            # Parse ISO 8601 timestamps from Alpaca API
            # issues with Pandas to_datetime anda had to change to ISO format! works!
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
            
            all_dfs.append(df)
            
            page += 1
            total_rows += len(df)
            
            if show_progress:
                print(f"  {symbol} | page {page} | rows this page: {len(df)} | total: {total_rows:,}")
            
            # Check for next page
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            
            params["page_token"] = next_token
            
            # Small delay to be nice to the API
            time.sleep(0.1)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.set_index('timestamp', inplace=True)
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.sort_index(inplace=True)
            return combined_df
        
        return pd.DataFrame()
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            symbol: Stock symbol
            timeframe: Timeframe string
            
        Returns:
            Path to saved file
        """
        filepath = self._get_parquet_path(symbol, timeframe)
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        print(f"  Saved {len(df):,} rows to {filepath}")
        return filepath
    
    def load_from_parquet(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load data from parquet file.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        filepath = self._get_parquet_path(symbol, timeframe)
        
        if filepath.exists():
            df = pd.read_parquet(filepath)
            print(f"  Loaded {len(df):,} rows from {filepath}")
            return df
        
        return None
    
    def fetch_and_save(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch data and save to parquet. Load from cache if available.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            force_refresh: Force re-download even if cache exists
            
        Returns:
            DataFrame with historical data
        """
        if not force_refresh:
            cached_df = self.load_from_parquet(symbol, timeframe)
            if cached_df is not None:
                return cached_df
        
        print(f"  Downloading {symbol} from {start_date} to {end_date} ({timeframe})...")
        df = self.fetch_symbol(symbol, start_date, end_date, timeframe)
        
        if not df.empty:
            self.save_to_parquet(df, symbol, timeframe)
        else:
            print(f"  Warning: No data returned for {symbol}")
        
        return df
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            force_refresh: Force re-download
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(symbols)}] Processing {symbol}")
            print(f"{'='*60}")
            
            df = self.fetch_and_save(
                symbol, start_date, end_date, timeframe, force_refresh
            )
            results[symbol] = df
            # počakam - delay
            if i < len(symbols):
                time.sleep(0.5)
        
        return results
    
    def get_available_data(self) -> Dict[str, dict]:
        """
        Get information about available cached data.
        
        Returns:
            Dictionary with file info
        """
        info = {}
        
        for filepath in self.data_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(filepath)
                parts = filepath.stem.split("_")
                symbol = parts[0]
                timeframe = "_".join(parts[1:])
                
                info[filepath.stem] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": len(df),
                    "start": df.index.min() if len(df) > 0 else None,
                    "end": df.index.max() if len(df) > 0 else None,
                    "size_mb": filepath.stat().st_size / (1024 * 1024)
                }
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        return info


def main():
    """Main function to download all historical data."""
    from config import SYMBOLS, START_DATE, END_DATE, DEFAULT_TIMEFRAME
    
    fetcher = AlpacaDataFetcher()
    
    print("Starting historical data download...")
    print(f"Symbols: {SYMBOLS}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Timeframe: {DEFAULT_TIMEFRAME}")
    print()
    
    results = fetcher.fetch_multiple_symbols(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=DEFAULT_TIMEFRAME,
        force_refresh=False
    )
    
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    
    for symbol, df in results.items():
        if not df.empty:
            print(f"{symbol}: {len(df):,} rows from {df.index.min()} to {df.index.max()}")
        else:
            print(f"{symbol}: No data")
    
    print("\nCached files:")
    for name, info in fetcher.get_available_data().items():
        print(f"  {name}: {info['rows']:,} rows, {info['size_mb']:.2f} MB")


if __name__ == "__main__":
    main()