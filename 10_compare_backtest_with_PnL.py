# Compate
### Backtest PnL - on  historical signals
### Live PnL - from actual IB paper trades stored in SQLLite3 (on TK machine)



import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("market_data.db")

df = pd.read_sql("""
SELECT date, close
FROM ohlcv
WHERE symbol = 'AAPL'
ORDER BY date
""", conn, parse_dates=['date'])

df.set_index('date', inplace=True)


df['SMA20'] = df['close'].rolling(20).mean()
df['SMA50'] = df['close'].rolling(50).mean()
df = df[df['SMA50'].notna()]


print(df)

## generate backtestss

trades_bt = []

position = 0
entry_price = 0

for i in range(1, len(df)):
    prev = df.iloc[i-1]
    curr = df.iloc[i]

    buy = prev['SMA20'] <= prev['SMA50'] and curr['SMA20'] > curr['SMA50']
    sell = prev['SMA20'] >= prev['SMA50'] and curr['SMA20'] < curr['SMA50']

    if buy and position == 0:
        position = 1
        entry_price = curr['close']
        entry_date = curr.name

    elif sell and position == 1:
        exit_price = curr['close']
        exit_date = curr.name

        pnl = exit_price - entry_price
        # tale append še ne dela, ker moram nafilati actaul trades
        trades_bt.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl
        })
        position = 0

bt_df = pd.DataFrame(trades_bt)

#povyetek backtest

bt_total_pnl = bt_df['pnl'].sum()
bt_trades = len(bt_df)

print("______BAKCTEST______")
print(bt_total_pnl)
print(bt_trades)

 
