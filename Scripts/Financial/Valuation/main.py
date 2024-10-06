'''
new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/Financial/dark box'  # Replace with the path to your desired directory
os.chdir(new_directory)
'''

#LIBRERIAS
from datetime import date
import pandas as pd
import plotly.io as pio
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
pio.renderers.default = 'browser'

#MODULES
from Scripts.Financial.Database.ticker_scrape import price,companydescription,av_financials
from Scripts.Financial.Valuation.valuation import damodaran_2
from Scripts.Financial.Database.macro_scrape import dmd,gdpworld,fred

# FUNCIONES
def tickerdata(ticker):
    key = 'B6T9Z1KKTBKA2I1C'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    financial_statements = av_financials(ticker, key, headers) #alphavantage 2009-today quaterly frecuency

    #fin = financialdata(ticker) #yahoo last 4 years
    start_date = financial_statements['Date'].iloc[-1]


    end_date = date.today()
    ticker_data={'description': companydescription(ticker), 'financial_statements': financial_statements, 'price': price(ticker, start_date - pd.DateOffset(years=6), end_date)}
    return ticker_data,start_date

def macrodata(start_date):

    SPY = price('SPY',start_date - pd.DateOffset(years=6), date.today())

    rates=fred()['rf'].mean()
    ERP = dmd()[['Year', 'Implied ERP (FCFE)']]['Implied ERP (FCFE)'].mean()
    g=gdpworld()
    g,g_std=g['gdp growth'].iloc[-1],g['gdp growth'].std()

    macros = {'SPY':SPY,'Rf':rates,'ERP':ERP,'g':g,'g_std':g_std}

    return macros

def results_plotter(ticker_data,results):
    fig = px.line(ticker_data['price'], x='Date', y='Adj Close')  # Existing line

    results['Date_t0']
    results['TarjetPrice_t0']
    results['TarjetPrice_t1']
    results['R2']
    results['Fitting function']
    results['Fitting params']
    results['Projected Financial Statements']

    # Extract necessary data from ticker_data['description']
    ticker_info = ticker_data['description']
    summary = ticker_info['Summary']  # Exclude the summary
    link_yf = ticker_info['link_yf']
    target_mean_price = ticker_info['targetMeanPrice']
    target_high_price = ticker_info['targetHighPrice']
    target_low_price = ticker_info['targetLowPrice']

    # Show the plot
    fig.show()

    return


if __name__ == "__main__":

    ticker = 'AAPL'
    #for ticker in tickerlist(calculate return

        # GET STOCK AND MACROECONOMICAL VARIABLES
        ticker_data,start_date=tickerdata(ticker)
        macros=macrodata(start_date)

        # VALUATE STOCK
        results=damodaran_2(ticker_data,macros)


        # SHOW RESULTS
        results_plotter(ticker_data,results)


    '''
     plot grpah with price taarjet(x2) + yf reference, margin errors, r2and low plot with
    company interest rate is dependant on rf or rfund
    how to project beta
    
    '''