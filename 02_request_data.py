from ib_insync import IB, Stock, util

util.startLoop()  

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)


contract = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(contract)

ticker = ib.reqMktData(contract)
ib.sleep(2)

print("Symbol:", contract.symbol)
print("Bid:", ticker.bid)
print("Ask:", ticker.ask)
print("Last:", ticker.last)
print("Close:", ticker.close)

ib.disconnect()