"""
Daily updater for Alpaca stock data.
Fetches recent bars for AAPL and MSFT and inserts only new rows
into bars_daily and bars_hourly — safe to run multiple times per day.

Requirements:
    pip install alpaca-py pandas python-dotenv
    (sqlite3 is built into Python — no extra install needed)

Usage:
    python3 update_alpaca_data.py

Tip: schedule with cron to run automatically, e.g. every 4 hours:
    0 */4 * * * /usr/bin/python3 /path/to/update_alpaca_data.py

    
run:
cd 01_Data
python3 03_update_daily_data.py

"""

import os
import sqlite3
from typing import Optional
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
DB_PATH    = Path("alpaca_data.db")

# How far back to look on each run (overlap ensures no gaps are missed)
LOOKBACK_DAYS_DAILY  = 7    # fetch last 7 days of daily bars
LOOKBACK_DAYS_HOURLY = 3    # fetch last 3 days of hourly bars

NOW       = datetime.now(tz=timezone.utc)
END_DATE  = NOW.replace(minute=0, second=0, microsecond=0)   # current hour

RESOLUTIONS = {
    "daily":  (TimeFrame(1, TimeFrameUnit.Day),  LOOKBACK_DAYS_DAILY),
    "hourly": (TimeFrame(1, TimeFrameUnit.Hour), LOOKBACK_DAYS_HOURLY),
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def latest_timestamp(conn: sqlite3.Connection, table: str, symbol: str) -> Optional[str]:
    """Return the most recent timestamp stored for a symbol, or None."""
    cur = conn.cursor()
    cur.execute(
        f"SELECT MAX(timestamp) FROM {table} WHERE symbol = ?", (symbol,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def insert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> int:
    """
    Insert rows using INSERT OR IGNORE — duplicates are silently skipped.
    Returns the number of new rows inserted.
    """
    cols = ["symbol", "timestamp", "open", "high", "low",
            "close", "volume", "trade_count", "vwap"]
    cols = [c for c in cols if c in df.columns]

    rows         = [tuple(row[c] for c in cols) for _, row in df.iterrows()]
    placeholders = ", ".join(["?"] * len(cols))
    col_names    = ", ".join(cols)
    sql          = (f"INSERT OR IGNORE INTO {table} ({col_names}) "
                    f"VALUES ({placeholders})")

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Alpaca client
# ---------------------------------------------------------------------------
client = StockHistoricalDataClient(API_KEY, API_SECRET)


# ---------------------------------------------------------------------------
# Fetch & update one symbol / resolution
# ---------------------------------------------------------------------------
def update(conn: sqlite3.Connection,
           symbol: str,
           label: str,
           timeframe: TimeFrame,
           lookback_days: int) -> None:

    table      = f"bars_{label}"
    start_date = END_DATE - timedelta(days=lookback_days)

    # Check what we already have
    last_stored = latest_timestamp(conn, table, symbol)

    if last_stored:
        # Start just after the last stored bar to minimise data transfer
        last_dt    = datetime.fromisoformat(last_stored).replace(tzinfo=timezone.utc)
        start_date = max(start_date, last_dt + timedelta(seconds=1))
        status_msg = f"last stored: {last_stored[:16]} UTC"
    else:
        status_msg = "no existing data — full lookback fetch"

    print(f"  {symbol} [{label}]  {status_msg}  fetching from {start_date.strftime('%Y-%m-%d %H:%M')} UTC …")

    if start_date >= END_DATE:
        print(f" Already up to date, nothing to fetch")
        return

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_date,
        end=END_DATE,
        feed="iex",       # free feed; switch to "sip" on paid plan
        adjustment="all", # split- and dividend-adjusted
    )

    bars = client.get_stock_bars(request)
    df: pd.DataFrame = bars.df

    if df.empty:
        print(f" No new bars available yet")
        return

    df = df.reset_index()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )

    inserted = insert_df(conn, df, table)
    skipped  = len(df) - inserted
    print(f"  {len(df):,} bars fetched  →  "
          f"{inserted:,} inserted, {skipped:,} already existed")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    run_time = NOW.strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 60)
    print(f"  Alpaca Data Updater  —  {run_time}")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"  Database not found at {DB_PATH.resolve()}")
        print("   Run fetch_alpaca_data.py first to create and populate the database.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        for symbol in SYMBOLS:
            print(f"\n Symbol: {symbol}")
            print(f" {'─' * 38}")
            for label, (timeframe, lookback) in RESOLUTIONS.items():
                update(conn, symbol, label, timeframe, lookback)

    print(f"\n Update complete  →  {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()