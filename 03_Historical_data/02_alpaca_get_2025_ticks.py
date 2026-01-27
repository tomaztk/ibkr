import requests
import os
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Config
# -----------------------
HEADERS = {
    "accept": "application/json",
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET"),
}



""""
import requests

url = "https://data.alpaca.markets/v1beta1/options/bars?symbols=ESZ25&timeframe=1Min&start=2025-01-01T00%3A00%3A00Z&end=2026-01-22T00%3A00%3A00Z&limit=1000&sort=asc"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
print(response.text)
"""
BASE_URL = "https://data.alpaca.markets/v1beta1/options/bars"

https://data.alpaca.markets/v2/futures/bars?symbols=ESZ25&timeframe=1Min&start=2025-01-01T00%3A00%3A00Z&end=2026-01-25T07%3A59%3A30.160553%2B00%3A00&limit=1000&sort=asc

SYMBOLS = ["ESZ25", "NQZ25", "GCZ25"]
TIMEFRAME = "1Min"
START = "2025-01-01T00:00:00Z"
END = datetime.now(timezone.utc).isoformat()

OUTPUT_DIR = "04_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_symbol(symbol: str) -> pd.DataFrame:
    params = {
        "symbols": symbol,   # ONE symbol only
        "timeframe": TIMEFRAME,
        "start": START,
        "end": END,
        "limit": 1000,
        "sort": "asc",
    }

    dfs = []
    page = 0

    while True:
        r = requests.get(BASE_URL, headers=HEADERS, params=params)
        r.raise_for_status()
        payload = r.json()

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
            "n": "trades",
            "vw": "vwap",
        })

        df["symbol"] = symbol
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        dfs.append(df)

        page += 1
        print(f"{symbol} | page {page} | rows {len(df)}")

        next_token = payload.get("next_page_token")
        if not next_token:
            break

        params["page_token"] = next_token

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()



for symbol in SYMBOLS:
    print(f"\nFetching {symbol}...")
    df = fetch_symbol(symbol)

    if df.empty:
        print(f"No data for {symbol}")
        continue

    out_path = f"{OUTPUT_DIR}/{symbol}_1min_2025_to_today.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
