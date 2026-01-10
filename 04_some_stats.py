from ib_insync import IB, Stock, util
import pandas as pd
import numpy as np
import mplfinance as mpf

# ---------------------------
# Connect to IB
# ---------------------------
util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=5)

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='3 M',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True
)

ib.disconnect()

df = util.df(bars)


### data
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df['SMA20'] = df['close'].rolling(20).mean()
df['SMA50'] = df['close'].rolling(50).mean()


# rsi14

delta = df['close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))



# MACD (12, 26, 9)
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()

df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']


# Buysells

df['signal'] = 0
df.loc[(df['SMA20'] > df['SMA50']) & (df['RSI'] < 70), 'signal'] = 1
df.loc[(df['SMA20'] < df['SMA50']) & (df['RSI'] > 30), 'signal'] = -1

df['buy'] = np.where(df['signal'] == 1, df['close'], np.nan)
df['sell'] = np.where(df['signal'] == -1, df['close'], np.nan)


df = df[['open', 'high', 'low', 'close', 'volume']].copy()
df.columns = df.columns.str.lower()
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

df = df[df['SMA50'].notna()]
df = df[df['SMA20'].notna()]
df = df[df['buy'].notna()]

df = df[df['sell'].notna()]

# Plot 

apds = [
    mpf.make_addplot(df['SMA20']),
    mpf.make_addplot(df['SMA50']),
    mpf.make_addplot(df['buy'], type='scatter', marker='^', markersize=100),
    mpf.make_addplot(df['sell'], type='scatter', marker='v', markersize=100)
]

mpf.plot(
    df,
    type='candle',
    addplot=apds,
    volume=True,
    style='charles'
)

