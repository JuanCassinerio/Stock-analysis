import yfinance as yf
import pandas as pd

from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
######################### COMPANY DATA SCRAPING
######################### DATA SCRAPING
def financialdata(ticker): #FINANCIAL STATEMENTS yahoo
    data_ticker = yf.Ticker(ticker)
    cf = data_ticker.cashflow.T.rename_axis('Date').reset_index()
    it = data_ticker.income_stmt.T.rename_axis('Date').reset_index()
    bs = data_ticker.balance_sheet.T.rename_axis('Date').reset_index()
    it = it.drop(columns='Date')
    bs = bs.drop(columns='Date')
    data = pd.concat([cf, it, bs], axis=1)
    return data

def price(ticker,start_date,end_date): #STOCKS PRICES (DIVIDEND ACCOUNTED)
    ticker='AAPL'
    price = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    price = pd.DataFrame(price)
    price['Date'] = price.index
    price['ticker'] = ticker
    price = price.sort_values(by='Date', ascending=False)
    price = price.reset_index(drop=True)
    return price

SPY =price('SPY', start_date, end_date)
SPY = SPY .reset_index(drop=True)

def companydescription(ticker): # COMPANY DESCRIPTION
    company_info = yf.Ticker(ticker).info
    currency = company_info.get("currency", None)
    shortName = company_info.get("shortName", None)
    industry = company_info.get("industry", None)
    sector = company_info.get("sector", None)
    country = company_info.get("country", None)
    longBusinessSummary = company_info.get("longBusinessSummary", None)
    link_yf = f"https://finance.yahoo.com/quote/{ticker}?.tsrc=fin-srch"
    new_row = {
        "ticker": ticker,
        "currency": currency,
        "Company": shortName,
        "industry": industry,
        "sector": sector,
        "country": country,
        "Summary": longBusinessSummary,
        "link_yf": link_yf,
    }
    return new_row



'''
ALPHAVANTAGE https://www.alphavantage.co  KEY 'B6T9Z1KKTBKA2I1C'
'''

import requests
import pandas as pd

def av_financials(ticker,key,headers):

    sections = ['BALANCE_SHEET', 'INCOME_STATEMENT', 'CASH_FLOW']

    result_df = pd.DataFrame()
    all_sections_data = True
    for section in sections:
        url = f'https://www.alphavantage.co/query?function={section}&symbol={ticker}&apikey={key}'
        r = requests.get(url, headers=headers)

        #check for ticker info availability
        if r.status_code == 200:  # Check if request is successful
            data = r.json()
            if len(data) == 1: #200 yes but maximum request reached
                all_sections_data = False
                break #breaks all for loops, upt to the main for loop

            # checks all sections
            if "quarterlyReports" in data:  # Check if quarterlyReports exist in data
                quarterly_reports = data["quarterlyReports"]
                data = pd.DataFrame.from_dict(quarterly_reports)
                if result_df.empty:
                    result_df = data.copy()
                else:

                    result_df = pd.merge(result_df, data, on='fiscalDateEnding', how='left')
            else:
                all_sections_data = False
                print(f"Not complete quarterly reports for {ticker}")
                continue
        else:
            print(f"Fail request to API. Status code: {r.status_code}")

    if all_sections_data:
        print(f"{ticker} - Succesfull request for 3 financial statements branches")
        result_df = result_df.dropna()
        result_df['ticker'] = ticker
        result_df = result_df.rename(columns={'fiscalDateEnding': 'Date'})




        fin=ticker_data['financial_statements']



        fin = fin.rename(columns={'fiscalDateEnding': 'Date',
                                  'cashAndCashEquivalentsAtCarryingValue': 'Cash',
                                  'commonStockSharesOutstanding': 'Shares',
                                  'totalAssets': 'Assets',
                                  'capitalExpenditures': 'PPE',
                                  'totalAssets': 'Net Income',
                                  'totalAssets': 'Revenue',
                                  'totalAssets': 'Depreciation',
                                  'totalAssets': 'Assets Current',
                                  'totalAssets': 'Liabilities Current',
                                  'totalAssets': 'Liabilities',
                                  'totalAssets': 'Date',
                                  'totalAssets': 'Date',
                                  'totalAssets': 'Date',
                                  'totalAssets': 'Date',
                                  'totalAssets': 'Date',
                                  'totalAssets': 'Date',
                                  'totalAssets': 'Date',})
        fin['Date'] = pd.to_datetime(fin['Date'])
        for col in ['Shares', 'totalNonCurrentLiabilities', 'Cash', 'operatingCashflow', 'PPE',
                    'incomeTaxExpense', 'incomeBeforeTax', 'interestExpense']:
            fin[col] = pd.to_numeric(fin[col], errors='coerce')

        fin['tax'] = fin['incomeTaxExpense'] / fin['incomeBeforeTax']
        reversed_tax_ratio = fin['tax'].iloc[::-1]
        fin['tax'] = reversed_tax_ratio.rolling(window=16, min_periods=16).median().iloc[::-1]  # mean
        reversed_interest_expense = fin['interestExpense'].iloc[::-1]
        fin['interestExpensecumm'] = reversed_interest_expense.rolling(window=4, min_periods=4).sum().iloc[::-1]
        fin['kd'] = fin['interestExpensecumm'] / fin['totalNonCurrentLiabilities']




        fin2 = fin2[['Date', 'fcf', 'Shares', 'totalNonCurrentLiabilities', 'Cash', 'kd', 'tax']]



    return result_df









