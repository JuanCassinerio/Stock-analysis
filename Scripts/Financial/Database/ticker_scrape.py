import yfinance as yf
import pandas as pd
import requests


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
    price = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    price = pd.DataFrame(price)
    price = price.sort_values(by='Date', ascending=False)
    price['Date'] = price.index
    price['ticker'] = ticker
    price = price.reset_index(drop=True)
    return price


def companydescription(ticker): # COMPANY DESCRIPTION

    company_info = yf.Ticker(ticker).info
    currency = company_info.get("currency", None)
    shortName = company_info.get("shortName", None)
    industry = company_info.get("industry", None)
    sector = company_info.get("sector", None)
    country = company_info.get("country", None)
    longBusinessSummary = company_info.get("longBusinessSummary", None)
    targetHighPrice = company_info.get('targetHighPrice', None)
    targetLowPrice=company_info.get('targetLowPrice', None)
    targetMeanPrice=company_info.get('targetMeanPrice', None)

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
        "targetHighPrice" : targetHighPrice,
        "targetLowPrice" : targetLowPrice,
        "targetMeanPrice" : targetMeanPrice,
    }
    return new_row



'''
ALPHAVANTAGE https://www.alphavantage.co  KEY 'B6T9Z1KKTBKA2I1C'
'''


def av_financials(ticker,key,headers):

    '''

    ticker='AAPL'
    key = 'B6T9Z1KKTBKA2I1C'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    '''

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




    #DATA PREPARATION
        #VARIABLES AND FORMAT
        result_df = result_df.rename(columns={'fiscalDateEnding': 'Date',
                                  'totalRevenue': 'Revenue',
                                  'netIncomeFromContinuingOperations': 'Net Income',
                                  'depreciationAndAmortization': 'Depreciation',
                                  'capitalExpenditures': 'Capex',
                                  'propertyPlantEquipment': 'PPE',
                                  'cashAndShortTermInvestments': 'Cash and ST Investments',
                                  'cashAndCashEquivalentsAtCarryingValue': 'Cash',
                                  'totalCurrentAssets': 'Assets Current',
                                  'totalAssets': 'Assets',
                                  'totalCurrentLiabilities': 'Liabilities Current',
                                  'totalLiabilities': 'Liabilities',
                                  'longTermDebtNoncurrent': 'Long Term Debt',
                                   'commonStockSharesOutstanding': 'Shares'})
        result_df['Date'] = pd.to_datetime(result_df['Date'])
        for col in ['Revenue','Net Income','Depreciation','Capex','PPE','Cash and ST Investments',
                    'Cash','Assets Current', 'Assets','Liabilities Current','Liabilities','Long Term Debt', 'Shares',
                    'incomeTaxExpense','incomeBeforeTax','interestExpense','totalNonCurrentLiabilities']:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

        #tax
        result_df['tax'] = result_df['incomeTaxExpense'] / result_df['incomeBeforeTax']
        reversed_tax_ratio = result_df['tax'].iloc[::-1]
        result_df['tax'] = reversed_tax_ratio.rolling(window=16, min_periods=16).median().iloc[::-1]  # mean
        #kd
        reversed_interest_expense = result_df['interestExpense'].iloc[::-1]
        result_df['interestExpensecumm'] = reversed_interest_expense.rolling(window=4, min_periods=4).sum().iloc[::-1]
        result_df['kd'] = result_df['interestExpensecumm'] / result_df['totalNonCurrentLiabilities']

        result_df = result_df[['Date','Revenue','Net Income','Depreciation','Capex','PPE','Cash and ST Investments',
                               'Cash','Assets Current', 'Assets','Liabilities Current','Liabilities','Long Term Debt',
                               'Shares','kd', 'tax']]
    return result_df


def sec():
    headers = {'User-Agent': "juancassinerio@gmail.com"}
    companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json",headers=headers)
    companyData = pd.DataFrame.from_dict(companyTickers.json(),orient='index')
    companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10) #tickers
    n=50 #first 50 tickers by marketcap
    tickers = companyData.iloc[:n]['ticker'].tolist() #ranked by marketap dsc
    return tickers




