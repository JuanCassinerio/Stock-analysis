import pandas as pd
import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import yfinance as yf
from plotly.offline import iplot
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests
import re
import warnings
import pandas_datareader as pdr
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

import warnings

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


tickers = ['ABT', 'HWM', 'AEG', 'MO', 'AOCA', 'AEM', 'AMX', 'AXP', 'AIG', 'AMGN','ADI', 'AAPL', 'AMAT', 'T', 'ADP', 'AVY', 'CAR', 'NTCO', 'AZN', 'BBVA','BBD', 'BSBR', 'BAC', 'BCS', 'GOLD', 'BAS GR', 'BAYN GR', 'BHP', 'BA','BP', 'BMY', 'BG', 'CAH', 'CAT', 'CX', 'CVX', 'CSCO', 'C', 'KO', 'KOF','CDE', 'CL', 'VALE', 'GLW', 'COST', 'MBG GR', 'DE', 'DTEA GR', 'DEO','DIS', 'DD', 'EBAY', 'EOAN GR', 'LLY', 'E', 'XOM', 'FNMA', 'FDX', 'FSLR','FMX', 'ORAN', 'FMCC', 'FCX', 'GE', 'GSK', 'GFI', 'GOOGL', 'PAC', 'ASR','TV', 'BSN GR', 'HOG', 'HMY', 'HDB', 'HL', 'HPQ', 'HMC', 'HHPD LI', 'HON','HSBC', 'HNPIY', 'IBM', 'IBN', 'INFY', 'ING', 'INTC', 'IFF', 'IP', 'JPM','JNJ', 'KMB', 'KGC', 'KB', 'KEP', 'LVS', 'LYG', 'ERIC', 'LMT', 'MMC','PCRFY', 'MCD', 'MDT', 'MELI', 'MRK', 'MSFT', 'MMM', 'MUFG', 'MFG', 'MBT','MSI', 'NGG', 'NEC1 GR', 'NEM', 'NKE', 'NSANY', 'NOK', 'NMR', 'NVS', 'NUE','OGZD', 'LKOD', 'ATAD', 'NLMK LI', 'ORCL', 'PCAR', 'PSO', 'PEP', 'PTRCY','PFE', 'PHG', 'PBI', 'PKX', 'PG', 'QCOM', 'BB', 'RIO', 'SHEL', 'SMSN LI','SAP', 'SLB', 'SIEGY', 'SNA', 'SONY', 'SCCO', 'SBUX', 'SYY', 'TSM', 'TTM','TIIAY', 'TEF', 'TX', 'TXN', 'BK', 'HSY', 'HD', 'TTE', 'TM', 'TRV', 'JCI','USB', 'UL', 'X', 'RTX', 'VZ', 'VOD', 'WMT', 'WFC', 'AABA', 'YZCAY', 'META','AMZN', 'NVDA', 'ADBE', 'BIIB', 'GILD', 'NFLX', 'TMO', 'PYPL', 'CRM', 'TSLA','AMD', 'NG', 'TRIP', 'ANF', 'URBN', 'GRMN', 'SNAP', 'VRSN', 'XRX', 'YELP','ROST', 'TGT', 'GS', 'V', 'ARCO', 'DESP', 'AGRO', 'GLOB', 'BABA', 'BIDU','ABEV', 'VIV', 'JD', 'NTES', 'TCOM', 'YY', 'GGB', 'BRFS', 'CBD', 'SBS','WB', 'ITUB', 'ERJ', 'UGP', 'SUZ', 'EBR', 'ELP', 'TIMB', 'SID', 'TS', 'SAN','PBR', 'VIST', 'ABBV', 'BRKB', 'BIOX', 'AVGO', 'CAAP', 'DOCU', 'ETSY', 'GPRK','HAL', 'MA', 'PAAS', 'PSX', 'UNP', 'UNH', 'WBA', 'ZM', 'EFX', 'SQ', 'SHOP','SPOT', 'SNOW', 'TWLO', 'COIN', 'SPGI', 'AAL', 'LRCX', 'EA', 'XP', 'GM','DOW', 'AKO.B', 'NIO', 'SE', 'ADS']
len(tickers)

y=5 #Years

start_date = datetime.datetime.now() - datetime.timedelta(days=365*y)
end_date = datetime.datetime.now()
data = yf.download(tickers, start=start_date, end=end_date)
data = data["Adj Close"]
data = data.dropna(axis=1)
common_tickers = data.columns.tolist()


def func(x, a, b, c):
     return a*((b)**(x-c) +1)

n = 1  # days to exclude
results = []

for ticker in common_tickers:
    data.index = data[ticker].index.astype('int64')
    scaler = MinMaxScaler()
    data_index_scaled = scaler.fit_transform(data.index.values.reshape(-1, 1)).flatten()

    popt, _ = curve_fit(func, data_index_scaled[:-n], data[ticker][:-n], maxfev=100000)
    trend_data = func(data_index_scaled, *popt)
    r2 = r2_score(data[ticker][:-n], trend_data[:-n])
    v = ((data[ticker] - trend_data) * 100 / trend_data).std
    derivative = popt[0] * (popt[1]**(data_index_scaled[-1]- popt[2]) * np.log(popt[1]))

    # Check if the slope is positive
    if r2 > 0.8 and derivative > 0:
        p = ((data[ticker].iloc[-1] - trend_data[-1]) / trend_data[-1]) * 100

        if p < 0:
            status = 'Below Trend'
        else:
            status = 'Above Trend'

        if p < -v() :
            vstatus = 'Strong Buy'

        elif p < -2*v() :
            vstatus = 'Very Strong Buy'
        else:
            vstatus = '-'

        results.append({'Ticker': ticker, 'Percentage Difference': p, 'Current Status': status, "Volatility": v(),'V Status': vstatus,})

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Percentage Difference')
results_df

ticker = "IBN"
def func(x, a, b,c):
    return a*((b)**(x-c) +1)
#     return a*x**2+b*x+c

y=5
start_date = datetime.datetime.now() - datetime.timedelta(days=365 * y)
end_date = datetime.datetime.now()
df = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
df = pd.DataFrame(df)
df['Date'] = df.index
df.index = df.index.astype('int64')

scaler = MinMaxScaler()
df_index_scaled = scaler.fit_transform(df.index.values.reshape(-1, 1)).flatten()

n=1
popt, _ = curve_fit(func, df_index_scaled[:-n], df['Adj Close'][:-n], maxfev=100000)
trend_data = func(df_index_scaled, *popt)
r2 = r2_score(df['Adj Close'][:-n], trend_data[:-n])

v=((df['Adj Close'] - trend_data) * 100 / trend_data).std

fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{ticker}', 'Percentage Difference %'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], name=f'{ticker} Stock Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=trend_data, name='Trend'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=(df['Adj Close'] - trend_data) * 100 / trend_data, name='Cycle + Error (%)', line=dict(color='blue')), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date'], y=[0] * len(df.index), mode='lines', name='Zero Line', line=dict(color='gray', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date'], y=[0] * len(df.index), mode='lines', name='Zero Line', line=dict(color='gray', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date'], y=[2* v()]  * len(df.index), mode='lines', name='Upper Limit', line=dict(color='red', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date'], y=[-2 * v()] * len(df.index), mode='lines', name='Lower Limit', line=dict(color='green', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date'], y=[v()] * len(df.index), mode='lines', name='Upper Threshold', line=dict(color='orange', dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date'], y=[-v()] * len(df.index), mode='lines', name='Lower Threshold', line=dict(color='lightgreen', dash='dash')), row=1, col=2)
fig.update_layout(xaxis_title='Date', yaxis_title=f'Price ({ticker})', height=450, width=1800)
fig.show()

r2

tickers = results_df.iloc[:, 0].tolist()
common_tickers = tickers[:10]
n = len(common_tickers)
start_date = datetime.datetime.now() - datetime.timedelta(days=365*y)
end_date = datetime.datetime.now()
df = yf.download(common_tickers, start=start_date, end=end_date)['Adj Close']
num_rows = 2
num_cols = (n + 1) // 2
fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=common_tickers[:n])
for i, ticker in enumerate(common_tickers[:n]):
    row = (i // num_cols) + 1
    col = (i % num_cols) + 1
    fig.add_trace(go.Scatter(x=df.index, y=df[ticker], mode='lines', name=ticker), row=row, col=col)
fig.update_layout(height=800, width=1800, title_text=f"Stock Price Data for {n} Common Tickers in {num_rows} Rows")
fig.show()
