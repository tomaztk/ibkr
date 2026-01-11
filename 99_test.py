
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


df['SMA20'] = df['close'].rolling(10).mean()
df['SMA50'] = df['close'].rolling(20).mean()
df = df[df[['SMA20']].notna().any(axis=1)]
df = df[df[['SMA50']].notna().any(axis=1)]

latest = df.iloc[-1]
previous = df.iloc[-2]


buy_signal = ( previous['SMA20'] <= previous['SMA50'] and latest['SMA20'] > latest['SMA50'] )
sell_signal = ( previous['SMA20'] >= previous['SMA50'] and latest['SMA20'] < latest['SMA50'] )

print(buy_signal)
print(sell_signal)
 
 

from ib_insync import IB, Stock, MarketOrder, util

util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)

# Give IB time to sync
ib.sleep(5)

summary = ib.accountSummary()
print("Account OK, items:", len(summary))

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)


print(contract)


#satniy check - position
positions = ib.positions()
current_position = sum(p.position for p in positions if p.contract.symbol == 'AAPL')

print("Current AAPL position:", current_position)


## paper order placing

QTY = 10  # change carefully

#test
order = MarketOrder('BUY', QTY)
trade = ib.placeOrder(contract, order)
ib.sleep(1)
print(trade)


''' 
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
'''
ib.disconnect()
