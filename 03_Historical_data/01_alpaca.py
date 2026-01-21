# Alpaca

import requests
import os
from dotenv import load_dotenv
import polars as pl

load_dotenv()


headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET")
}

#feed=sip
url = (
    "https://data.alpaca.markets/v2/stocks/bars"
    "?symbols=AAPL"
    "&timeframe=1Min"
    "&limit=1000"
    "&adjustment=raw"
    "&feed=iex"
    "&sort=asc"
)

#print("KEY:", os.getenv("ALPACA_API_KEY"))
#print("SECRET:", os.getenv("ALPACA_API_SECRET"))

response = requests.get(url, headers=headers)

print(response.text)

data = response.text

bars = data["bars"]

dfs = []

for symbol, rows in bars.items():
    df = (
        pl.DataFrame(rows)
        .rename({
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "trades",
            "vw": "vwap",
        })
        .with_columns([
            pl.lit(symbol).alias("symbol"),
            pl.col("timestamp").str.strptime(
                pl.Datetime,
                format="%Y-%m-%dT%H:%M:%SZ",
                strict=False
            )
        ])
        .select([
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trades",
            "vwap",
        ])
    )
    dfs.append(df)

bars_df = pl.concat(dfs)
