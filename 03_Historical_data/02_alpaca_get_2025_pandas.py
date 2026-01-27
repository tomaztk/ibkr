import requests
import os
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    "accept": "application/json",
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET"),
}

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

params = {
    "symbols": "AAPL",
    "timeframe": "1Min",
    "start": "2025-01-01T00:00:00Z",
    "end": datetime.now(timezone.utc).isoformat(),
    "limit": 1000,
    "adjustment": "raw",
    "feed": "iex",  # use iex unless you have SIP
    "sort": "asc",
}

dfs = []
page = 0

while True:
    response = requests.get(BASE_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    payload = response.json()

    bars = payload.get("bars", {}).get("AAPL", [])
    if not bars:
        break

    df = pd.DataFrame(bars)

    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trades",
        "vw": "vwap",
    })

    df["symbol"] = "AAPL"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    dfs.append(df)

    page += 1
    print(f"Fetched page {page}, rows: {len(df)}")

    # pagination
    next_token = payload.get("next_page_token")
    if not next_token:
        break

    params["page_token"] = next_token

bars_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print(bars_df.shape)
print(bars_df.head())

bars_df.to_parquet("04_Data/AAPL_1min_2025_to_today.parquet", index=False)
