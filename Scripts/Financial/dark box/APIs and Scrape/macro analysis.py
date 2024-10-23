"""
!pip install pandas_datareader
!pip install fredapi
"""

import pandas as pd
import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import yfinance as yf
from plotly.offline import iplot
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import re
import warnings
import pandas_datareader.data as web
from fredapi import Fred
import pandas_datareader as pdr
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

start_date = datetime.datetime.now() - datetime.timedelta(days=365*10)
end_date = datetime.datetime.now()

# https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL + personal account api key => 1ad73d818bcf7a9313ed3bac1802f40b
api_key ='1ad73d818bcf7a9313ed3bac1802f40b'
fred = Fred(api_key=api_key)
m = fred.get_series("CPILFESL", start_date=start_date, end_date=end_date)  # Retrieve the cpi


start_date = datetime.datetime.now() - datetime.timedelta(days=365*10)
end_date = datetime.datetime.now()

# https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL + personal account api key => 1ad73d818bcf7a9313ed3bac1802f40b
api_key ='1ad73d818bcf7a9313ed3bac1802f40b'
fred = Fred(api_key=api_key)
inflation = pd.DataFrame(fred.get_series('CORESTICKM159SFRBATL',start_date,end_date))  #inflation usa / add this line to select the first column of the DataFrame
inflation = inflation.iloc[:, 0]
fedfunds = web.DataReader("FEDFUNDS", "fred", start_date, end_date)  #federal fund rate usa(night)
m = fred.get_series("BOGMBASE", start_date=start_date, end_date=end_date)  # Retrieve the M0 Money Stock data
df = pd.DataFrame({'FEDFUNDS': fedfunds['FEDFUNDS'], 'inflation': inflation, 'M0': m})
df = df.resample('M').mean()
df['Month'] = df.index.strftime('%Y-%m')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.dropna()

tickers = ["^MERV","^GSPC", "^IXIC", "^DJI", "GD=F", "000001.SS", "GC=F", "CL=F", "BTC-USD", "^N225", "DAX"] # Define the tickers and the date range
data = yf.download(tickers, start=start_date, end=end_date)

fig1 = make_subplots(rows=3, cols=3, subplot_titles=("S&P 500", "NASDAQ Composite Index", "Dow Jones", "Shangai Composite", "DAX", "Nikkei", "Merval", "BTC", "Commodity Index"))
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["^GSPC"]), row=1, col=1)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["^IXIC"]), row=1, col=2)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["^DJI"]), row=1, col=3)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["000001.SS"]), row=2, col=1)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["DAX"]), row=2, col=2)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["^N225"]), row=2, col=3)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["^MERV"]), row=3, col=1)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["BTC-USD"]), row=3, col=2)
fig1.add_trace(go.Scatter(x=data.index, y=data["Adj Close"]["GD=F"]), row=3, col=3)
fig1.update_layout(title="Financial Index", showlegend=False)
fig1.show()

fig2 = make_subplots(rows=1, cols=3, subplot_titles=("M2 Money Stock", "Inflation", "Federal Fund Rate"))
fig2.add_trace(go.Scatter(x=df['Month'], y=df['M0']), row=1, col=1)
fig2.add_trace(go.Scatter(x=df['Month'], y=df['inflation']), row=1, col=2)
fig2.add_trace(go.Scatter(x=df['Month'], y=df['FEDFUNDS']), row=1, col=3)
fig2.update_layout(title="Economic Indicators", showlegend=False)
fig2.show()



start_date = datetime.datetime.now() - datetime.timedelta(days=365*10)
end_date = datetime.datetime.now()

# https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL + personal account api key => 1ad73d818bcf7a9313ed3bac1802f40b
api_key ='1ad73d818bcf7a9313ed3bac1802f40b'
fred = Fred(api_key=api_key)
inflation = pd.DataFrame(fred.get_series('CORESTICKM159SFRBATL',start_date,end_date))  #inflation usa / add this line to select the first column of the DataFrame
inflation = inflation.iloc[:, 0]
fedfunds = web.DataReader("FEDFUNDS", "fred", start_date, end_date)  #federal fund rate usa
m = fred.get_series("BOGMBASE", start_date=start_date, end_date=end_date)  # Retrieve the M0 Money Stock data
df = pd.DataFrame({'FEDFUNDS': fedfunds['FEDFUNDS'], 'inflation': inflation, 'M0': m})
df = df.resample('M').mean()
df['Month'] = df.index.strftime('%Y-%m')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.dropna()




import pandas as pd
import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import yfinance as yf
from plotly.offline import iplot
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats as stats
from fredapi import Fred
import pandas_datareader.data as web


api_key = '1ad73d818bcf7a9313ed3bac1802f40b'
fred = Fred(api_key=api_key)

tickers = ["PCOALAUUSDM", "PNGASUSUSDM", "DCOILBRENTEU", "PWHEAMTUSDM", "PSOYBUSDQ", "PIORECRUSDM", "PCOPPUSDM", "PRUBBUSDM"]
y=3
start_date = datetime.datetime.now() - datetime.timedelta(days=365*y)
end_date = datetime.datetime.now()

fig = make_subplots(rows=2, cols=4, subplot_titles=["Coal", "Gas", "Oil", "Wheat", "Soybean", "Iron", "Copper", "Rubber"])
normalized_data= pd.DataFrame()
for i, ticker in enumerate(tickers):
    data = fred.get_series(ticker, start_date=start_date, end_date=end_date)
    data = data.loc[data.index >= start_date]  # Filter data based on start date
    normalized_data[ticker] = data / data.iloc[0]
    row = (i // 4) + 1
    col = (i % 4) + 1
    fig.add_trace(go.Scatter(x=data.index, y=data, name=ticker), row=row, col=col)

    fig.update_yaxes(title_text="Price [$]", row=row, col=col)
    fig.update_xaxes(title_text="Date", row=row, col=col)

fig.update_layout(height=800, width=2000, title="Commodity Prices")
fig.show()


normalized_data = pd.DataFrame()

for ticker in tickers:
    data = fred.get_series(ticker, start_date=start_date, end_date=end_date)
    data = data.loc[data.index >= start_date]  # Filter data based on start date
    normalized_data[ticker] = data / data.iloc[0]

sum_normalized_data = normalized_data.sum(axis=1)

## Commodity Index S&P-GSCI Commodity Index Future
ci = yf.download("GD=F", start=start_date, end=end_date)["Adj Close"]
ci=ci/ci.iloc[0]
cix= sum_normalized_data/len(tickers )
fig = go.Figure()
fig.add_trace(go.Scatter(x=ci.index, y=ci, name='Commodity Index S&P-GSCI'))
fig.add_trace(go.Scatter(x=cix.index, y=cix, name='Mannual one'))
fig.show()























