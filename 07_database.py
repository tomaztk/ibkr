## Local SQLlite3

import sqlite3
from ib_insync import IB, Stock, util
import pandas as pd
import numpy as np


DB_FILE = "market_data.db"

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

'''
### tableca crate open-high-low-close-volume
cursor.execute("""
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
)
""")
conn.commit()
'''


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



### insert from IBKR to sqllite3 :)

df_db = df.reset_index().copy()
df_db['symbol'] = 'AAPL'

df_db = df_db[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
df_db['date'] = df_db['date'].dt.strftime('%Y-%m-%d')


rows = df_db.to_records(index=False)

cursor.executemany("""
INSERT OR IGNORE INTO ohlcv (symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)
""", rows)

conn.commit()



### Load back to pd
query = """
SELECT date, open, high, low, close, volume
FROM ohlcv
WHERE symbol = 'AAPL'
ORDER BY date
"""

df_sql = pd.read_sql(query, conn, parse_dates=['date'])
df_sql.set_index('date', inplace=True)
# simple count!
cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol='AAPL'")
print("Rows in DB:", cursor.fetchone()[0])