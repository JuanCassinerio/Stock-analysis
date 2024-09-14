'''
new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/varios/dark box'  # Replace with the path to your desired directory
os.chdir(new_directory)

'''

#LIBRERIAS

from datetime import date, timedelta
import plotly.io as pio
import plotly.express as px
import os

pio.renderers.default='browser'

#MODULES
from modules import tickerdata,price,companydescription
from damodaran import damodaran

# FUNCIONES

#
if __name__ == "__main__":

    # DATA SCRAPE
    start_date='2019-01-01'
    end_date=date.today()
    ticker='AAPL'

    financial_statements=tickerdata(ticker)
    price=price(ticker,start_date,end_date)
    description=companydescription(ticker)
    '''fig = px.line(price,x='Date', y='Adj Close', title=ticker)
    fig.show()'''

