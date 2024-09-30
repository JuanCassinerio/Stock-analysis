'''
new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/varios/dark box'  # Replace with the path to your desired directory
os.chdir(new_directory)
'''

#LIBRERIAS
from datetime import date
import pandas as pd
import plotly.io as pio
from pathlib import Path

pio.renderers.default='browser'

#MODULES
from Scripts.varios.Database.ticker_scrape import price,companydescription,av_financials
from valuation import damodaran_2
from

# FUNCIONES
def tickerdata(ticker):
    key = 'B6T9Z1KKTBKA2I1C'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    #financial_statements = av_financials(ticker, key, headers) #alphavantage 2009-today quaterly frecuency
    path = Path.cwd().parent.parent.absolute()/'Database'/'db'

    financial_statements = pd.read_csv(path/'financials.csv')
    #fin = financialdata(ticker) #yahoo last 4 years
    start_date = financial_statements['Date'].iloc[-1]


    end_date = date.today()
    ticker_data={'description': companydescription(ticker), 'financial_statements': financial_statements, 'price': price(ticker, start_date - pd.DateOffset(years=6), end_date)}
    return ticker_data,start_date

def macrodata(start_date):
    path = Path.cwd().parent.parent.absolute()/'Database'/'db'
    SPY = price('SPY',start_date - pd.DateOffset(years=6), date.today())

    rates=fred()
    ERP = MarketPremium()
    g=gdpworld()

    macros = {'SPY':SPY,'Rf':Rf,'ERP':ERP,'g':g}

    return macros

#
if __name__ == "__main__":



    ticker = 'AAPL'
    #for ticker in tickerlist(calculate return
        ticker_data,start_date=tickerdata(ticker)[0][1]


        macros={'Rf':4,'SPY':SPY,'g':3,'inflation':2.5,'Rp':ERP['Implied ERP (FCFE)'].iloc[-1]*100}

        macros=macrodata(start_date)

        price=damodaran(ticker_data,macros)[0]

        results = {'Date_t0': Datelast_date, 'TarjetPrice_t0': TarjetPrice_0today, 'TarjetPrice_t1': TarjetPrice_1yplus
            , 'R2': max_r2, 'Fitting function': best_function, 'Fitting params': best_fit_params
            , 'Projected Financial Statements': data}


        plot grpah with price taarjet(x2) + yf reference, margin errors, r2and low plot with


        if tp>p








    '''
    path = Path.cwd().absolute()/'Scripts'/'varios'/'Database'/'db'

    financial_statements = pd.read_csv(path/'financials.csv')
    financial_statements['Date'] = pd.to_datetime(financial_statements['Date'])
    
    start_date = financial_statements['Date'].iloc[-1]
    end_date = date.today()
    SPY=price('SPY', start_date, end_date)
   
 

    
    '''


    '''
    company interest rate is dependant on rf or rfund
    
    
    
    
    
    '''




