"""
Fetch historical OHLCV data for AAPL and MSFT from Alpaca Markets.
Stores results in a SQLite3 database with two tables: bars_daily, bars_hourly.

Requirements:
    pip install alpaca-py pandas python-dotenv
    (sqlite3 is built into Python — no extra install needed)

Database schema (same for both tables):
    id          INTEGER  PRIMARY KEY AUTOINCREMENT
    symbol      TEXT     e.g. 'AAPL'
    timestamp   TEXT     ISO-8601 UTC  e.g. '2024-01-15T14:30:00+00:00'
    open        REAL
    high        REAL
    low         REAL
    close       REAL
    volume      REAL
    trade_count REAL
    vwap        REAL
"""

import os
import sqlite3
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

END_DATE   = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=5 * 3)   # ~5 years

RESOLUTIONS = {
    "daily":  TimeFrame(1, TimeFrameUnit.Day),
    "hourly": TimeFrame(1, TimeFrameUnit.Hour),
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    trade_count REAL,
    vwap        REAL,
    UNIQUE(symbol, timestamp)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_{table}_symbol_ts
ON {table} (symbol, timestamp);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they don't exist yet."""
    cur = conn.cursor()
    for table in ("bars_daily", "bars_hourly"):
        cur.executescript(
            CREATE_TABLE_SQL.format(table=table) +
            CREATE_INDEX_SQL.format(table=table)
        )
    conn.commit()
    print("✓ Database initialised:", DB_PATH.resolve())


def insert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> int:
    """
    Insert rows into the given table.
    Uses INSERT OR IGNORE so re-runs are safe (no duplicates).
    Returns number of rows actually inserted.
    """
    cols = ["symbol", "timestamp", "open", "high", "low", "close",
            "volume", "trade_count", "vwap"]

    # Keep only columns that exist in the dataframe
    cols = [c for c in cols if c in df.columns]

    rows = [
        tuple(row[c] for c in cols)
        for _, row in df.iterrows()
    ]

    placeholders = ", ".join(["?"] * len(cols))
    col_names    = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# Alpaca client
# ---------------------------------------------------------------------------
client = StockHistoricalDataClient(API_KEY, API_SECRET)


# ---------------------------------------------------------------------------
# Fetch & store
# ---------------------------------------------------------------------------
def fetch_and_store(conn: sqlite3.Connection,
                    symbol: str,
                    label: str,
                    timeframe: TimeFrame) -> None:

    table = f"bars_{label}"   # bars_daily  or  bars_hourly
    print(f"  Fetching {symbol} [{label}] …")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=START_DATE,
        end=END_DATE,
        feed="iex",        # free feed; switch to "sip" on paid plan
        adjustment="all",  # split- and dividend-adjusted
    )

    bars = client.get_stock_bars(request)
    df: pd.DataFrame = bars.df

    if df.empty:
        print(f"    WARNING: no data returned for {symbol} [{label}]")
        return

    # Flatten multi-index (symbol, timestamp) → flat columns
    df = df.reset_index()

    # Normalise timestamp to ISO string (SQLite stores as TEXT)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        )

    inserted = insert_df(conn, df, table)
    print(f"    ✓ {symbol} [{label}]: {len(df):,} rows fetched, "
          f"{inserted:,} new rows inserted → table '{table}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Alpaca → SQLite3 Data Downloader")
    print("=" * 60)

    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)

        for symbol in SYMBOLS:
            print(f"\n{'─' * 40}")
            print(f" Symbol: {symbol}")
            print(f"{'─' * 40}")
            for label, timeframe in RESOLUTIONS.items():
                fetch_and_store(conn, symbol, label, timeframe)

        # Quick summary
        print(f"\n{'─' * 40}")
        print(" Database summary")
        print(f"{'─' * 40}")
        cur = conn.cursor()
        for table in ("bars_daily", "bars_hourly"):
            cur.execute(
                f"SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp) "
                f"FROM {table} GROUP BY symbol ORDER BY symbol"
            )
            for row in cur.fetchall():
                symbol, count, ts_min, ts_max = row
                print(f"  {table:<14} {symbol}  {count:>6,} rows  "
                      f"{ts_min[:10]} → {ts_max[:10]}")

    print(f"\nDone. Database saved to: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()