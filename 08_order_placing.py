# crossover signal detection
# and
# placing a new order?!

import sqlite3
import pandas as pd

conn = sqlite3.connect("market_data.db")

df = pd.read_sql("""
SELECT date, open, high, low, close, volume
FROM ohlcv
WHERE symbol = 'AAPL'
ORDER BY date
""", conn, parse_dates=['date'])

conn.close()

df.set_index('date', inplace=True)




### add SMA20 and SMA50 from 04_some_stats file! 
# or create a class!?!?!?!?!!?

# df['date'] = pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)
df['SMA20'] = df['close'].rolling(20).mean()
df['SMA50'] = df['close'].rolling(50).mean()

df = df[df[['SMA20']].notna().any(axis=1)]
df = df[df[['SMA50']].notna().any(axis=1)]


latest = df.iloc[-1]
previous = df.iloc[-2]

buy_signal = (
    previous['SMA20'] <= previous['SMA50']
    and latest['SMA20'] > latest['SMA50']
)

sell_signal = (
    previous['SMA20'] >= previous['SMA50']
    and latest['SMA20'] < latest['SMA50']
)

print("Buy signal:", buy_signal)
print("Sell signal:", sell_signal)


### create new class

from ib_insync import IB, Stock, MarketOrder, util

util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)

# a-sync; give smt time
ib.sleep(5)

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)


#satniy check - position
positions = ib.positions()
current_position = sum(p.position for p in positions if p.contract.symbol == 'AAPL')

print("Current AAPL position:", current_position)


## paper order placing

QTY = 10  # change accorgindly!

if buy_signal and current_position == 0:
    print("BUY signal → placing order")
    order = MarketOrder('BUY', QTY)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    print(trade)

elif sell_signal and current_position > 0:
    print("SELL signal → closing position")
    order = MarketOrder('SELL', current_position) ## marketOrder f() -> check the position
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    print(trade)

else:
    print("No trade action taken")

ib.disconnect()
