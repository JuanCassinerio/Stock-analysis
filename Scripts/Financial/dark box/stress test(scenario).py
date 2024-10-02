"""
Stress test/ Scenario Analysis
"""

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm




fed
spy, DataFrame
mlr regression

+ 3 cases, historically seen on dataframe, worst cases




#portfolio and data to use for analysis
portfolio_value = 1
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']
weights = np.array([1/len(tickers)] * len(tickers))

years = 15
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365*years)

adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=startDate, end=endDate)
    adj_close_df[ticker] = data['Adj Close']

#portfolio past behaviour

log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()
historical_returns = (log_returns * weights).sum(axis=1) #portfolio historical log returns

cov_matrix = log_returns.cov() * 252 
portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights) #volatility


+

risk free rate
sp500
gdp

correlation between this variables change and portfolio return

place scenarios and obain expected return(bad scenarios set of 3x3 combinations)

#2
'''
calculate weight(or portfolio that maximize sharpe, minimize volatility and maximize return)
'''