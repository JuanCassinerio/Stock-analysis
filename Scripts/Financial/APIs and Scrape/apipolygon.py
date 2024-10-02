import pandas as pd

import requests
"""
https://polygon.io/
"""

key="dNMRRoOQJdnEDTXpyp9xTcVfqcSJoChR"

# 5 calls per minute(all statementes) 10 yearly and 10 quaterly(last 2.5 years), better than yfinance, worst than macrotresnds, worst than alphavantage
# but "limitless" requests


url=f"https://api.polygon.io/vX/reference/financials?ticker=AAPL&timeframe=quarterly&apiKey=dNMRRoOQJdnEDTXpyp9xTcVfqcSJoChR"


url=f"https://api.polygon.io/vX/reference/financials?ticker=AAPL&timeframe=annual&apiKey=dNMRRoOQJdnEDTXpyp9xTcVfqcSJoChR"

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
data_json = requests.get(url, headers=headers, verify=False).json()
df = pd.DataFrame(data_json['results']) 
