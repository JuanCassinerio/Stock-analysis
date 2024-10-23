import pandas as pd
import datetime
import plotly.graph_objs as go
import yfinance as yf
from plotly.offline import iplot
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import curve_fit
import requests
import re
from bs4 import BeautifulSoup
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

'''
1 - Acciones
'''

tickers = ['GGAL.BA', 'ALUA.BA']

ticker=tickers[0]

data = yf.Ticker(ticker)
hdf = pd.DataFrame()
hfd = data.income_stmt.append(data.balancesheet)
hfd = hfd.T
hfd.index.name = 'Date'
hfd = hfd[::-1]
hfd = hfd.reset_index()  #estados financieros

starttime='2002-01-01'
endtime=datetime.datetime.today().strftime('%Y-%m-%d')
stock = yf.download(ticker, start=starttime, end=endtime)['Adj Close']  #precio accion

g=3

wacc
riesgo pais(damodaranm scrape)

dolar ccl(alphacast)

def damodaran(ticker,g):
    # Financial Data
    data = yf.Ticker(ticker)
    hdf = pd.DataFrame()
    hfd = data.income_stmt.append(data.balancesheet)
    hfd = hfd.T
    hfd.index.name = 'Date'
    hfd = hfd[::-1]
    hfd = hfd.reset_index()

    # Revenues Projection
    def func(t, a1):
      return hfd['Total Revenue'].iloc[0]*(1+a1/100)**(t-hfd['Year'].iloc[0])

    hfd['Year'] = hfd['Date'].dt.year
    popt, _ = curve_fit(func, hfd['Year'], hfd['Total Revenue'], maxfev=10000000)
    a1 = popt[0]

    # Vertical Analysis to Revenue
    variables = ['Net Income','Reconciled Depreciation','Net PPE','Current Assets','Total Non Current Assets', 'Current Liabilities','Total Non Current Liabilities Net Minority Interest','Common Stock Equity', 'Cash And Cash Equivalents']
    verticalratio = {}
    for variable in variables:
        ratio_name = f'{variable} Vertical Ratio'
        hfd[ratio_name] = hfd[variable] / hfd['Total Revenue']
        verticalratio[variable] = hfd[ratio_name].mean()

    # Other Variables(wacc - sharesOutstanding - Years_Depreciation)

    url = f"https://valueinvesting.io/{ticker}/valuation/wacc"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}  # Mimic a real browser to avoid potential blocking
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser") #this contains the info
    wacc_element = soup.find("meta", {"name": "description"})
    wacc_content = wacc_element["content"]
    wacc_value_start = wacc_content.find("The WACC of") + len("The WACC of")
    wacc_value_end = wacc_content.find("%", wacc_value_start)

    wacc= float(re.search(r"([\d.]+)", wacc_content[wacc_value_start:wacc_value_end].strip()).group(1))

    wacc=8.2

    shares = data.info['sharesOutstanding']
    Years_Depreciation=(hfd['Net PPE']/hfd['Reconciled Depreciation']).mean()
    Net_Debt=(hfd['Current Liabilities']+hfd['Total Non Current Liabilities Net Minority Interest']-hfd['Cash And Cash Equivalents']).iloc[-1]

    # Projection
    Datelast_num = hfd['Date'].dt.year.iloc[-1]
    Datelast_date = hfd['Date'].iloc[-1]
    Yearlast= hfd['Year'].iloc[-1]
    for i in range(1, 6):
        Date = Datelast_date + pd.DateOffset(years=1)*i
        Year = Yearlast + i
        Revenue = func(Year, a1)
        NetIncome = Revenue * verticalratio['Net Income']
        CurrentAssets = Revenue * verticalratio['Current Assets']
        NetIncome = Revenue * verticalratio['Net Income']
        CurrentAssets = Revenue * verticalratio['Current Assets']
        CurrentLiabilities = Revenue * verticalratio['Current Liabilities']
        Cash = Revenue * verticalratio['Cash And Cash Equivalents']
        TotalNonCurrentAssets = Revenue * verticalratio['Total Non Current Assets']
        TotalNonCurrentLiabilities = Revenue * verticalratio['Total Non Current Liabilities Net Minority Interest']
        NetPP = Revenue * verticalratio['Net PPE']
        Depreciation = NetPP/Years_Depreciation

        new_year_data = {'Date': Date,'Total Revenue':Revenue,'Net Income': NetIncome, 'Current Assets': CurrentAssets,'Current Liabilities': CurrentLiabilities,'Cash And Cash Equivalents': Cash,  'Total Non Current Assets':
                        TotalNonCurrentAssets, 'Total Non Current Liabilities Net Minority Interest': TotalNonCurrentLiabilities, 'Net PPE': NetPP,'Reconciled Depreciation': Depreciation}

        hfd = hfd.append(new_year_data, ignore_index=True)

    # FCFF
    Operatingcashflow = hfd['Net Income'] + hfd['Reconciled Depreciation']
    Capex = hfd['Net PPE'] - hfd['Net PPE'].shift(1) + hfd['Reconciled Depreciation']
    NWCCh = (hfd['Current Assets']-hfd['Current Liabilities']-hfd['Cash And Cash Equivalents']) - (hfd['Current Assets']-hfd['Current Liabilities']-hfd['Cash And Cash Equivalents']).shift(1)
    hfd['Free Cash Flow'] = Operatingcashflow - Capex - NWCCh

    fcfnext = hfd['Free Cash Flow'].iloc[-1] * (1+g/100)
    terminalvalue = fcfnext / ((wacc/100)-(g/100))
    Subtotal = hfd['Free Cash Flow'].tolist()

    # Modify the last element of the list by adding terminalvalue
    Subtotal[-1] += terminalvalue

    def npv(cash_flows, wacc, g):
        npv = 0
        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + (wacc/100)) ** t
        return npv

    VA_Asset = npv(Subtotal[-5:], wacc,g)
    VA_Equity=VA_Asset-Net_Debt
    TarjetPrice_mean = VA_Equity/shares

    return TarjetPrice_mean,wacc,hfd,shares,a1,verticalratio['Net Income']#growth rate


'''
ticker = 'nvda'
g = 3  # Economy Growth

stock_data = yf.download(ticker, start='2018-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))

current_price = stock_data['Close'][-1]
target_price= damodaran(ticker, g)[0]
action_text = "Buy" if current_price < target_price else "Sell"
action_color = "green" if current_price < target_price else "red"

# Plotting the graph with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price'))
fig.add_trace(go.Scatter(x=[datetime.datetime.today()], y=[target_price], mode='markers',marker=dict(color=action_color, size=15), name='Target Price'))
fig.add_annotation(x=datetime.datetime.today(),y=target_price,text=f'{action_text}',font=dict(color=action_color, size=30))
fig.update_layout(title=f'{ticker} Stock Price and Target Price Over Time',xaxis_title='Date',yaxis_title='Stock Price',)
print(ticker,"Tarjet Price is: $",round(damodaran(ticker,g)[0],0))
print(ticker,"wacc is: ",round(damodaran(ticker,g)[1],1),"%")

data = yf.Ticker(ticker)
print("Price of",f'{ticker} (365 days from today)',"= ",data.info['targetMeanPrice']," +/- ",(data.info['targetHighPrice']-data.info['targetLowPrice'])/2)
fig.show()
'''

'''
#data.info.items()
#data.major_holders
#data.institutional_holders
#data.mutualfund_holders
#data.news
#data.financials
#data.info['sector']
#data.info
#data.info['trailingPE']
#data.income_stmt
#data.balance_sheet
#data.cashflow
#data.quarterly_income_stmt
#data.quarterly_balance_sheet
#data.quarterly_cashflow
#historic financial data
#print('Amount of Years: ',hfd.shape[1])
'''

'''
2 - bonos y on
'''