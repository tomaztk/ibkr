"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("04_data")

dfs = []
for path in DATA_DIR.glob("*_1min_2025_to_today.parquet"):
    df = pd.read_parquet(path)
    dfs.append(df)

all_futures_df = pd.concat(dfs, ignore_index=True)

print(all_futures_df["symbol"].value_counts())

"""

"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("04_data")


# ETFs
etf_symbols = ["SPY", "QQQI"]
etf_dfs = [
    pd.read_parquet(DATA_DIR / f"{s}_1min_2025_to_today.parquet")
    for s in etf_symbols
]
etf_df = pd.concat(etf_dfs, ignore_index=True)
"""


# graphs


import pandas as pd
import mplfinance as mpf

spy = pd.read_parquet("04_data/SPY_1min_2025_to_today.parquet")

spy.head(5)

spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True)
spy = (
    spy
    .sort_values("timestamp")
    .set_index("timestamp")
)

spy_view = spy.loc["2025-01-01 20:30":"2025-01-01 22:00"]

 

mpf.plot(
    spy_view,
    type="candle",
    volume=True,
    style="yahoo",
    title="SPY · 1m · QQQ",
    ylabel="Price",
    ylabel_lower="Volume",
    datetime_format="%H:%M",
    tight_layout=True,
)

"""
es = pd.read_parquet("04_data/DIA_1min_2025_to_today.parquet")
es["timestamp"] = pd.to_datetime(es["timestamp"], utc=True)
es = es.set_index("timestamp").sort_index()

aligned = pd.concat(
    {
        "SPY": spy["close"],
        "DIA": es["close"]
    },
    axis=1
).loc["2025-01-01 20:30":"2025-01-01 22:00"]

aligned.plot(title="SPY vs DIA (1m)")
"""