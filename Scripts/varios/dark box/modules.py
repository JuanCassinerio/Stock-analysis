import yfinance as yf
import pandas as pd

from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
import requests
import re


def tickerdata(ticker):
    data_ticker = yf.Ticker(ticker)
    cf = data_ticker.cashflow.T.rename_axis('Date').reset_index()
    it = data_ticker.income_stmt.T.rename_axis('Date').reset_index()
    bs = data_ticker.balance_sheet.T.rename_axis('Date').reset_index()
    it = it.drop(columns='Date')
    bs = bs.drop(columns='Date')
    data = pd.concat([cf, it, bs], axis=1)
    return data

def price(ticker,start_date,end_date):
    price = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    price = pd.DataFrame(price)
    price['Date'] = price.index
    price['ticker'] = ticker

    return price


def companydescription(ticker):
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

# ALPHA

def salesprojection(x,a1,b1,c1,g,b2,c2):
    return a1*(1+b1/100)**(x-c1)+g/(1+np.exp(b2*(x-c2)))

def salesprojection(x,a,b):
    return a*x+b

def damodaran(data):
    data=financial_statements
    data['Year'] = data['Date'].dt.year
    data = data.dropna(subset=['Total Revenue'])

    popt, _ = curve_fit(salesprojection, data['Year'], data['Total Revenue'], maxfev=1000)
    slope = popt[0]
    intercept = popt[1]
    data.drop('Year', axis=1, inplace=True)

    # Vertical Analysis to Revenue
    variables = ['Net Income', 'Reconciled Depreciation', 'Net PPE', 'Current Assets', 'Total Non Current Assets',
                 'Current Liabilities', 'Total Non Current Liabilities Net Minority Interest',
                 'Cash And Cash Equivalents']
    verticalratio = {}
    for variable in variables:
        ratio_name = f'{variable} Vertical Ratio'
        data[ratio_name] = data[variable] / data['Total Revenue']
        verticalratio[variable] = data[ratio_name].mean()

    #wacc scraping
    url = f"https://valueinvesting.io/{ticker}/valuation/wacc"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}  # Mimic a real browser to avoid potential blocking
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")  # this contains the info
    wacc_element = soup.find("meta", {"name": "description"})
    wacc_content = wacc_element["content"]
    wacc_value_start = wacc_content.find("The WACC of") + len("The WACC of")
    wacc_value_end = wacc_content.find("%", wacc_value_start)
    wacc = float(re.search(r"([\d.]+)", wacc_content[wacc_value_start:wacc_value_end].strip()).group(1))

    #other variables
    Years_Depreciation = (data['Net PPE'] / data['Reconciled Depreciation']).mean()
    Net_Debt = (data['Current Liabilities'] + data['Total Non Current Liabilities Net Minority Interest'] - data[
        'Cash And Cash Equivalents']).iloc[-1]
    shares = data['Ordinary Shares Number'].iloc[-1]

    # Sales Projection
    Datelast_num = data['Date'].dt.year.iloc[-1]
    Datelast_date = data['Date'].iloc[-1]
    for i in range(1, 6):
        Date = Datelast_date + pd.DateOffset(years=1) * i
        Revenue = salesprojection(Datelast_num + i, slope, intercept)
        NetIncome = Revenue * verticalratio['Net Income']
        CurrentAssets = Revenue * verticalratio['Current Assets']
        CurrentLiabilities = Revenue * verticalratio['Current Liabilities']
        Cash = Revenue * verticalratio['Cash And Cash Equivalents']
        TotalNonCurrentAssets = Revenue * verticalratio['Total Non Current Assets']
        TotalNonCurrentLiabilities = Revenue * verticalratio['Total Non Current Liabilities Net Minority Interest']
        NetPP = Revenue * verticalratio['Net PPE']
        Depreciation = NetPP / Years_Depreciation

        new_year_data = {'Date': Date, 'Net Income': NetIncome, 'Current Assets': CurrentAssets,
                         'Current Liabilities': CurrentLiabilities, 'Cash And Cash Equivalents': Cash,
                         'Total Non Current Assets':
                             TotalNonCurrentAssets,
                         'Total Non Current Liabilities Net Minority Interest': TotalNonCurrentLiabilities,
                         'Net PPE': NetPP, 'Reconciled Depreciation': Depreciation}

        new_row_df = pd.DataFrame([new_year_data])

        # Use pd.concat() to add the new row to the existing DataFrame
        data = pd.concat([data, new_row_df], ignore_index=True)

    # FCFF
    Operatingcashflow = data['Net Income'] + data['Reconciled Depreciation']
    Capex = data['Net PPE'] - data['Net PPE'].shift(1) + data['Reconciled Depreciation']
    NWCCh = (data['Current Assets'] - data['Current Liabilities'] - data['Cash And Cash Equivalents']) - (
                data['Current Assets'] - data['Current Liabilities'] - data['Cash And Cash Equivalents']).shift(1)
    data['Free Cash Flow'] = Operatingcashflow - Capex - NWCCh

    g = 3

    fcfnext = data['Free Cash Flow'].iloc[-1] * (1 + g / 100)
    terminalvalue = fcfnext / ((wacc / 100) - (g / 100))
    Subtotal = data['Free Cash Flow'].tolist()
    Subtotal[-1] += terminalvalue

    def npv(cash_flows, wacc, g):
        npv = 0
        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + (wacc / 100)) ** t
        return npv

    VA_Asset = npv(Subtotal[-5:], wacc, g)
    VA_Equity = VA_Asset - Net_Debt
    TarjetPrice_mean = VA_Equity / shares
    return TarjetPrice_mean, wacc