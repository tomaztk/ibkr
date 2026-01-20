# Alpaca

import requests

# Credentials:
APCA-API-KEY-id
APCA-API-SECRETS

url = "https://data.alpaca.markets/v2/stocks/bars?limit=1000&adjustment=raw&feed=sip&sort=asc"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
print(response.text)