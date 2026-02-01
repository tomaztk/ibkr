
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

#print(buy_signal)
#print(sell_signal)
 
 

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


'''
9_test.py
False
False
Account OK, items: 71

Stock(conId=265598, symbol='AAPL', exchange='SMART', primaryExchange='NASDAQ', currency='USD',
 localSymbol='AAPL', tradingClass='NMS')

Current AAPL position: 0

Trade(contract=Stock(conId=265598, symbol='AAPL', exchange='SMART', primaryExchange='NASDAQ', 
currency='USD', localSymbol='AAPL', tradingClass='NMS'), order=MarketOrder(orderId=5, clientId=10, 
action='BUY', totalQuantity=10), orderStatus=OrderStatus(orderId=5, status='PendingSubmit', filled=0.0, 
remaining=0.0, avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', 
mktCapPrice=0.0), fills=[], log=[TradeLogEntry(time=datetime.datetime(2026, 1, 11, 6, 3, 17, 689015, 
tzinfo=datetime.timezone.utc), status='PendingSubmit', message='', errorCode=0)], advancedError='')

'''