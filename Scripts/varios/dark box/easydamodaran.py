#LIBRERIAS
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

# FUNCIONES

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

#
if __name__ == "__main__":

    start_date='2019-01-01'
    end_date=date.today()
    ticker='AAPL'

    financial_statements=tickerdata(ticker)
    price=plot_price(ticker,start_date,end_date)
    fig = px.line(price,x='Date', y='Adj Close', title=ticker)
    fig.show()