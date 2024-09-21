'''
Objective
.Use volume in trading startegy

Hyphotesis
.price change(absolute) vs volume transactions correlation
.high price change with low volume doesnt keep for long
.price change between 10-Q reports(strong data)

Application
.If volume is low consider a bull back to previous price
'''

import pandas as pd
import yfinance as yf
import datetime
import requests

import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'

headers = {'User-Agent': "juancassinerio@gmail.com"}
companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json",headers=headers)
companyData = pd.DataFrame.from_dict(companyTickers.json(),orient='index')
companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10) #tickers
n=500 #first 50 tickers by marketcap
tickers = companyData.iloc[:n]['ticker'].tolist()

starttime='2024-01-01'
endtime=datetime.datetime.today().strftime('%Y-%m-%d')

for ticker in tickes:
    data = yf.download(ticker, start=starttime, end=endtime)

    # Extract adjusted close price and volume
    adj_close = data['Adj Close']
    volume = data['Volume']
    
    # Calculate adjusted close price change
    adj_close_change = adj_close.diff()


#mean result parameter and graph




'''
- Conclusions
'''





