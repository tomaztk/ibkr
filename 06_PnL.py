import numpy as np
from ib_insync import IB, Stock, util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplfinance as mpf

 
## conn  
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

print(df)

df['SMA20'] = df['close'].rolling(20).mean()
df['SMA50'] = df['close'].rolling(50).mean()




delta = df['close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()

df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']
df['signal'] = 0
df.loc[(df['SMA20'] > df['SMA50']) & (df['RSI'] < 70), 'signal'] = 1
df.loc[(df['SMA20'] < df['SMA50']) & (df['RSI'] > 30), 'signal'] = -1
df['buy'] = np.where(df['signal'] == 1, df['close'], np.nan)
df['sell'] = np.where(df['signal'] == -1, df['close'], np.nan)



# -- end data

initial_capital = 100_000 #investmet
position = 0          # nof posicitons
cash = initial_capital
equity_curve = []

df['position'] = 0

for i in range(1, len(df)):
    price = df['close'].iloc[i]

    # buy if sma20 > sma50p ri loc = 0
    if df['SMA20'].iloc[i] > df['SMA50'].iloc[i] and position == 0:
        position = cash // price
        cash -= position * price
        df.iloc[i, df.columns.get_loc('position')] = position

    # sel if sma20<sma50 pri loc > 0 ???
    elif df['SMA20'].iloc[i] < df['SMA50'].iloc[i] and position > 0:
        cash += position * price
        position = 0
        df.iloc[i, df.columns.get_loc('position')] = 0

    equity = cash + position * price #equity 
    equity_curve.append(equity)

df = df.iloc[1:]
df['equity'] = equity_curve


# racaunam returnse
df['returns'] = df['equity'].pct_change()

total_return = (df['equity'].iloc[-1] / initial_capital - 1) * 100

# Drawdown :)
rolling_max = df['equity'].cummax()
drawdown = (df['equity'] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

print(f"Initial Capital: ${initial_capital:,.0f}")
print(f"Final Equity:    ${df['equity'].iloc[-1]:,.0f}")
print(f"Total Return:    {total_return:.2f}%")
print(f"Max Drawdown:    {max_drawdown:.2f}%")



plt.figure()
plt.plot(df.index, df['equity'])
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.show()
