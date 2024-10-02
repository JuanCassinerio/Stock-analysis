'''
https://www.zacks.com/stock/chart/AAPL/fundamental/beta

https://valueinvesting.io/AAPL/valuation/wacc
'''

import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


import pandas as pd
import yfinance as yf
import datetime
import numpy as np

file_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'
file_path1 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/price.csv'
file_path2 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/macro.csv'


tickers = pd.read_csv(file_path0)['ticker'].unique().tolist() + ['^GSPC']

price=pd.read_csv(file_path1)
ticker_already=price['ticker'].unique().tolist()

starttime='2002-01-01'
endtime=datetime.datetime.today().strftime('%Y-%m-%d')
latest_date = price['Date'].max()

macro = pd.read_csv(file_path2)  # '\t' is the tab separator
macro['Date'] = pd.to_datetime(macro['Date'])
macro['Month'] = macro['Date'].dt.month
macro['Year'] = macro['Date'].dt.year


'''
#add all info
for ticker in tickers:
    stock = yf.download(ticker, start=starttime, end=endtime)['Adj Close']
    if stock.size==0: #if no stock info
        continue
    stock = pd.DataFrame(stock)
    stock['Date'] = stock.index
    stock['ticker'] = ticker
    stock['Ke'] = np.nan
    stock = stock[['Date','Adj Close','Ke','ticker']]
    price = pd.concat([price,stock], ignore_index=True)
    
price.to_csv(file_path1, index=False)
'''


stock = yf.download("AAPL", start=pd.to_datetime(latest_date) + pd.Timedelta(days=1), end=endtime)['Adj Close']
stock = pd.DataFrame(stock)
stock['Date'] = stock.index
stock['Date'] = pd.to_datetime(stock['Date'], format='%Y-%m-%d')
starty=stock['Date'].min()


#add new dates
for ticker in tickers:
    stock = yf.download(ticker, start=starty, end=endtime)['Adj Close']

    if stock.size==0: #if no stock info
        continue
    stock = pd.DataFrame(stock)
    stock['Date'] = stock.index
    stock = stock.reset_index(drop=True)
    stock['ticker'] = ticker
    stock['Ke'] = np.nan
    stock = stock[['Date','Adj Close','Ke','ticker']]
    price = pd.concat([price,stock], ignore_index=True)
    
price['Date'] = pd.to_datetime(price['Date'])

price.to_csv(file_path1, index=False)


#ticker='AAPL'
#data = price.groupby('ticker').get_group(ticker)
#fig = px.line(data, x='Date', y='Adj Close', title='Adj Close vs Date')
#fig.show()


spy = price[price['ticker'] == '^GSPC']
spy = spy.drop('Ke', axis=1)


#calculate ke to each new stock date
for ticker in tickers:
    stock = price[price['ticker'] == ticker]
    
    not_ticker = price['ticker'] != ticker #drop ticker
    price = price[not_ticker]
    
    
    stock = stock.drop('Ke', axis=1)
    stock['Month'] = stock['Date'].dt.month
    stock['Year'] = stock['Date'].dt.year
    df = pd.merge(spy, stock, on='Date')
    df = df.rename(columns={'Adj Close_x': 'SPY', 'Adj Close_y': 'stock'})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.resample('M').ffill()
    df['SPY']=df['SPY'].pct_change()
    df['stock']=df['stock'].pct_change()
    df=df.dropna()
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    window = 60 # 5 x 12
    df['cov'] = df['SPY'].rolling(window=window).cov(df['stock'])
    df['var'] = df['SPY'].rolling(window=window).var()
    df['beta'] = df['cov'] / df['var']
    df = pd.merge(df, macro, on=['Year', 'Month'], how='inner')
    df['Ke']=df['RF']+df['beta']*df['MP']
    df.rename(columns={'Date_x': 'Date'}, inplace=True)
    df = df[['Year','Month','Ke']]
    df = pd.merge(stock, df, on=['Year', 'Month'], how='left')
    df = df[['Date','Adj Close','Ke','ticker']]
    
    price = pd.concat([price,df], ignore_index=True) #add ticker with Ke

price.to_csv(file_path1, index=False)










