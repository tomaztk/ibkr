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
        log_trade(trade, symbol='MBTH6')
        print("Trade logged")




### Place an order and log it

from ib_insync import IB, Stock, MarketOrder, util, Future

util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)
#print(ib)

#Unknown contract: Stock(symbol='MBT', exchange='MKT', currency='USD')

#contract = Stock('AAPL', 'SMART', 'USD')
#contract = Stock('MBTH6', 'MKT', 'USD')
contract = Future('MBTH6', 'MKT', 'USD')
ib.qualifyContracts(contract)

order = MarketOrder('BUY', 10)
trade = ib.placeOrder(contract, order)



# Wait for fill or status update
ib.sleep(2)

# Log trade
log_trade(trade, symbol='MBTH6')
# trade.filledEvent += on_filled


ib.disconnect()


'''
Error 10349, reqId 4: Order TIF was set to DAY based on order preset.
Canceled order: Trade(contract=Stock(conId=265598, symbol='AAPL', exchange='SMART', 
primaryExchange='NASDAQ', currency='USD', localSymbol='AAPL', tradingClass='NMS'), 
order=MarketOrder(orderId=4, clientId=12, action='BUY', totalQuantity=10), 
orderStatus=OrderStatus(orderId=4, status='Cancelled', filled=0.0, remaining=0.0, 
avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', 
mktCapPrice=0.0), fills=[], l
og=[TradeLogEntry(time=datetime.datetime(2026, 1, 11, 14, 34, 32, 659860, 
tzinfo=datetime.timezone.utc), status='PendingSubmit', message='', errorCode=0),
 TradeLogEntry(time=datetime.datetime(2026, 1, 11, 14, 34, 32, 686052, tzinfo=datetime.timezone.utc), 
 status='Cancelled', message='Error 10349, reqId 4: 
 Order TIF was set to DAY based on order preset.', errorCode=10349)], advancedError='')
'''


'''
Canceled order: Trade(contract=Stock(symbol='MBT', exchange=' MKT', currency='USD'), 
order=MarketOrder(orderId=16, clientId=12, action='BUY', totalQuantity=10), 
orderStatus=OrderStatus(orderId=16, status='Cancelled', filled=0.0, remaining=0.0, 
avgFillPrice=0.0, permId=0, parentId=0, lastFillPrice=0.0, clientId=0, whyHeld='', 
mktCapPrice=0.0), fills=[], log=[TradeLogEntry(time=datetime.datetime(2026, 1, 11, 17, 13, 42, 504215, 
tzinfo=datetime.timezone.utc), status='PendingSubmit', message='', errorCode=0), 
TradeLogEntry(time=datetime.datetime(2026, 1, 11, 17, 13, 42, 636949, tzinfo=datetime.timezone.utc), 
status='Cancelled', message='Error 200, reqId 16: The destination or exchange selected is Invalid. P
lease review your order\'s "Destination" field. If using a <br>Directed order, review the exchange 
selected when creating the order ticket or order row. This may occur when <br>creating stock orders 
for the overnight session or when creating option orders for the overnight session.', errorCode=200)], 
advancedError='')
'''