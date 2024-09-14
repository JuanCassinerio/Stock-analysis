import yfinance as yf
import pandas as pd

def tickerdata(ticker):
    data_ticker = yf.Ticker(ticker)
    cf = data_ticker.cashflow.T.rename_axis('Date').reset_index()
    it = data_ticker.income_stmt.T.rename_axis('Date').reset_index()
    bs = data_ticker.balance_sheet.T.rename_axis('Date').reset_index()
    it = it.drop(columns='Date')
    bs = bs.drop(columns='Date')
    data = pd.concat([cf, it, bs], axis=1)
    return data

def plot_price(ticker,start_date,end_date):
    price = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    price = pd.DataFrame(price)
    price['Date'] = price.index
    price['ticker'] = ticker

    return price