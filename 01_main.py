
## Connection test


from ib_insync import IB, util
from ib_insync import Stock
util.startLoop()  # needed for some environments (e.g. Jupyter)

 
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

print("Connected!")
print("Accounts:", ib.managedAccounts())

# ib.disconnect()

## working
## Connected!
## Accounts: ['DUO523270']



## Get market


contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

ticker = ib.reqMktData(contract)
ib.sleep(2)
print(ticker.last, ticker.bid, ticker.ask)

#print(contract)
ib.disconnect()


### get historical data

'''
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 M',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True
)

df = util.df(bars)
print(df.tail())

'''

### place an order

'''
order = MarketOrder('BUY', 10)
trade = ib.placeOrder(contract, order)

trade.filledEvent += lambda trade: print(trade)

'''