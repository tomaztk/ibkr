from ib_insync import IB, Stock, util
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf


util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=2)

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 Y',
   #  barSizeSetting='1 day',
    barSizeSetting='1 min',
    whatToShow='TRADES',
    useRTH=True
)

df = util.df(bars)
print(df[['date', 'open', 'high', 'low', 'close']])

plt.figure()
plt.title("AAPL - Daily OHLC")
plt.xlabel("Date")
plt.ylabel("Price")

for _, row in df.iterrows():
    plt.plot([row['date'], row['date']], [row['low'], row['high']])
    plt.plot(row['date'], row['open'], marker='_')
    plt.plot(row['date'], row['close'], marker='_')

plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
mpf.plot( df, type='candle', style='charles', title='AAPL - Daily Candlestick', volume=True)

ib.disconnect()
