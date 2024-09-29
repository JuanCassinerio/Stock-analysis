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
    ticker_data={'description': companydescription(ticker), 'financial_statements': financial_statements, 'price': price(ticker, start_date, end_date)}
    return ticker_data

def macrodata():
    SPY
    rates
    inflation=inflation()['inflation'].mean
    g=gdpworld()


    macros = {'description': companydescription(ticker), 'financial_statements': financial_statements,
                   'price': price(ticker, start_date, end_date)}

    return macros

#
if __name__ == "__main__":



    ticker = 'AAPL'
    #for ticker in tickerlist
    ticker_data=tickerdata(ticker)


    macros={'Rf':4,'SPY':SPY,'g':3,'inflation':2.5,'Rp':ERP['Implied ERP (FCFE)'].iloc[-1]*100}



    price=damodaran(ticker_data,macros)[0]




    '''
    path = Path.cwd().parent.parent.absolute()/'Database'/'db'

    financial_statements = pd.read_csv(path/'financials.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    start_date = financial_statements['Date'].iloc[-1]
    end_date = date.today()
    SPY=price('SPY', start_date, end_date)
   
 

    
    '''


    '''
    company interest rate is dependant on rf or rfund
    
    
    
    
    
    '''




