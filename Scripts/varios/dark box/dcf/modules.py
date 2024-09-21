import yfinance as yf
import pandas as pd

from scipy.optimize import curve_fit
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
######################### DATA SCRAPING

def financialdata(ticker): #FINANCIAL STATEMENTS
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
    price['Date'] = price.index
    price['ticker'] = ticker

    return price


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

######################### DATA SCRAPING

def salesprojection(x,a1,b1,c1,g,b2,c2):
    b1=2.5
    return a1*(1+b1/100)**(x-c1)+g/(1+np.exp(b2*(x-c2)))

def salesprojection(x,a,b):
    return a*x+b

def damodaran(ticker_data):
    ticker = ticker_data['description']['ticker']
    data = ticker_data['financial_statements']

    for col in data.columns:
        if col != 'Date':
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to float, coerce invalid to NaN


    data['Year'] = data['Date'].dt.year
    data = data.dropna(subset=['Total Revenue'])

    # project revenues (scaled data)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    years_scaled = scaler_x.fit_transform(data['Year'].values.reshape(-1, 1)).flatten()
    revenue_scaled = scaler_y.fit_transform(data['Total Revenue'].values.reshape(-1, 1)).flatten()
    popt, _ = curve_fit(salesprojection, years_scaled, revenue_scaled, maxfev=100)


    data = data.sort_values(by='Date', ascending=True)
    data['generated'] = 0


    data.drop('Year', axis=1)

    # Vertical Analysis to Revenue
    variables = ['Net Income', 'Reconciled Depreciation', 'Net PPE', 'Current Assets', 'Total Non Current Assets',
                 'Current Liabilities', 'Total Non Current Liabilities Net Minority Interest',
                 'Cash And Cash Equivalents']


    verticalratio = {variable: pd.DataFrame({variable: data[variable] / data['Total Revenue'] for variable in variables})[variable].mean() for variable in variables}


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

    # fcf Projection

    Datelast_date = data['Date'].iloc[-1]


    future_years = pd.date_range(start=Datelast_date + pd.DateOffset(years=1), periods=5, freq='YE')

    revenues = salesprojection(scaler_x(future_years.year.values.reshape(-1, 1)).flatten(), *popt)
    revenues = scaler_y.inverse_transform(revenues.reshape(-1, 1)).flatten() #unScale

    future_years = pd.date_range(start=Datelast_date + pd.DateOffset(years=1), periods=5, freq='YE')
    future_years_scaled = scaler_x.transform(future_years.year.values.reshape(-1, 1)).flatten()


    revenues = scaler_y.inverse_transform(salesprojection(future_years_scaled, *popt).reshape(-1, 1))

    net_incomes = revenues * verticalratio['Net Income']
    current_assets = revenues * verticalratio['Current Assets']
    current_liabilities = revenues * verticalratio['Current Liabilities']
    cash = revenues * verticalratio['Cash And Cash Equivalents']
    total_non_current_assets = revenues * verticalratio['Total Non Current Assets']
    total_non_current_liabilities = revenues * verticalratio['Total Non Current Liabilities Net Minority Interest']
    net_pp = revenues * verticalratio['Net PPE']
    depreciation = net_pp / Years_Depreciation

    revenues = revenues.flatten()
    net_incomes = net_incomes.flatten()
    current_assets = current_assets.flatten()
    current_liabilities = current_liabilities.flatten()
    cash = cash.flatten()
    total_non_current_assets = total_non_current_assets.flatten()
    total_non_current_liabilities = total_non_current_liabilities.flatten()
    net_pp = net_pp.flatten()
    depreciation = depreciation.flatten()

    # Create a DataFrame for new data
    new_year_data = {
        'Date': future_years,
        'Total Revenue': revenues,
        'Net Income': net_incomes,
        'Current Assets': current_assets,
        'Current Liabilities': current_liabilities,
        'Cash And Cash Equivalents': cash,
        'Total Non Current Assets': total_non_current_assets,
        'Total Non Current Liabilities Net Minority Interest': total_non_current_liabilities,
        'Net PPE': net_pp,
        'Reconciled Depreciation': depreciation,
        'generated': 1
    }

    new_df = pd.DataFrame(new_year_data)
    data = pd.concat([data, new_df], ignore_index=True)


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


    #extra analysis
    ''' return graprh

    data['color'] = data['generated'].apply(lambda x: 'red' if x == 0 else 'blue')
    fig = px.scatter(data,x='Date',y='Total Revenue', color='color')
    fig.update_traces(mode='markers+lines', line=dict(color='black', dash='dash'),marker=dict(size=10))
    fig.update_layout(title='Total Revenue')
    fig.show()

    
    
    '''

    return TarjetPrice_mean, wacc