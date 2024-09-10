import yfinance as yf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg'
import datetime

def plot_price(ticker)
    stock = yf.download(ticker, start=starttime, end=endtime)['Adj Close']

    stock = pd.DataFrame(stock)
    stock['Date'] = stock.index
    stock['ticker'] = ticker

    plt.figure(figsize=(10, 6))  # Set plot size
    plt.plot(stock['Date'], stock['Adj Close'])
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.title(f"{ticker} - Valores Históricos")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

