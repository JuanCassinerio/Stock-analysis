'''
new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/varios/dark box'  # Replace with the path to your desired directory
os.chdir(new_directory)
'''

#LIBRERIAS
from datetime import date, timedelta
import pandas as pd
import plotly.io as pio
import plotly.express as px
import os

pio.renderers.default='browser'

#MODULES
from modules import financialdata,price,companydescription,damodaran
#from damodaran import damodaran

# FUNCIONES
def tickerdata(ticker):
    financial_statements = financialdata(ticker)
    start_date = financial_statements['Date'].iloc[-1]
    end_date = date.today()
    ticker_data={'description': companydescription(ticker), 'financial_statements': financial_statements, 'price': price(ticker, start_date, end_date)}
    return ticker_data

#
if __name__ == "__main__":

    ticker = 'AAPL'
    #for ticker in tickerlist
    ticker_data=tickerdata(ticker)
    '''fig = px.line(price,x='Date', y='Adj Close', title=ticker)
        fig.show()'''

    price=damodaran(ticker_data)[0]