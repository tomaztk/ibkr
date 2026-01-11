import sqlite3

'''

conn = sqlite3.connect("market_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    symbol TEXT,
    action TEXT,
    quantity INTEGER,
    order_type TEXT,
    status TEXT,
    filled INTEGER,
    avg_fill_price REAL,
    order_id INTEGER,
    perm_id INTEGER
)
""")

conn.commit()
conn.close()

'''


from datetime import datetime
import sqlite3

def log_trade(trade, symbol):
    order = trade.order
    status = trade.orderStatus

    conn = sqlite3.connect("market_data.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO trades (
        timestamp,
        symbol,
        action,
        quantity,
        order_type,
        status,
        filled,
        avg_fill_price,
        order_id,
        perm_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        symbol,
        order.action,
        order.totalQuantity,
        order.orderType,
        status.status,
        status.filled,
        status.avgFillPrice,
        order.orderId,
        order.permId
    ))

    conn.commit()
    conn.close()


def on_filled(trade):
    if trade.orderStatus.status == 'Filled':
        log_trade(trade, symbol='AAPL')
        print("Trade logged")




### Place an order and log it

from ib_insync import IB, Stock, MarketOrder, util

util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=12)

contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

order = MarketOrder('BUY', 10)
trade = ib.placeOrder(contract, order)



# Wait for fill or status update
ib.sleep(2)

# Log trade
log_trade(trade, symbol='AAPL')
# trade.filledEvent += on_filled


ib.disconnect()
