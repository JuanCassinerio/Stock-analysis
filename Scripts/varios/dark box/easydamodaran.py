#LIBRERIAS

from datetime import date, timedelta
import plotly.io as pio
import plotly.express as px
import yfinance as yf
pio.renderers.default='browser'

# FUNCIONES

from modules import tickerdata


#
if __name__ == "__main__":

    start_date='2019-01-01'
    end_date=date.today()
    ticker='AAPL'

    financial_statements=tickerdata(ticker)
    price=plot_price(ticker,start_date,end_date)
    fig = px.line(price,x='Date', y='Adj Close', title=ticker)
    fig.show()