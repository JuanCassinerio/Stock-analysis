'''
ALPHAVANTAGE https://www.alphavantage.co  KEY 'B6T9Z1KKTBKA2I1C'

https://www.macrotrends.net/stocks/charts/AAPL/apple/net-income
'''

import requests
import pandas as pd

file_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'
key='B6T9Z1KKTBKA2I1C'


QCPI02FP1413CHTE
sections=['BALANCE_SHEET','INCOME_STATEMENT','CASH_FLOW']
fin=pd.read_csv(file_path0)
ticker_already = pd.read_csv(file_path0)['ticker'].unique().tolist()

headers = {'User-Agent': "juancassinerio@gmail.com"}
companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json",headers=headers)
companyData = pd.DataFrame.from_dict(companyTickers.json(),orient='index')
companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10) #tickers
n=500 #first 50 tickers by marketcap
tickers = companyData.iloc[:n]['ticker'].tolist()
tickers = [ticker for ticker in companyData.iloc[:n]['ticker'].tolist() if ticker not in ['SPY','LTMAY','CRM','FMX','GE','QQQ','EADSY','RTNTF']]



'''
stock = yf.download("AAPL", start=pd.to_datetime(latest_date) + pd.Timedelta(days=1), end=endtime)['Adj Close']
stock = pd.DataFrame(stock)
stock['Date'] = stock.index
stock['Date'] = pd.to_datetime(stock['Date'], format='%Y-%m-%d')
starty=stock['Date'].min()


#add new dates
for ticker in tickers:
    stock = yf.download(ticker, start=starty, end=endtime)['Adj Close']

    if stock.size==0: #if no stock info
        continue
'''

i=0
d=0
for ticker in tickers:
    if d==1:
        print("Exceeded for today")
        break
    if ticker not in ticker_already:
        print (ticker)
        continue
    all_sections_data = True
    result_df = pd.DataFrame()
    for section in sections:  
        url = f'https://www.alphavantage.co/query?function={section}&symbol={ticker}&apikey={key}'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:  # Check if request is successful
            data = r.json()
            if  len(data)==1:
                all_sections_data = False
                d=1
                break
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
            print(f"Fail request. Status code: {r.status_code}")

    if all_sections_data:
        result_df = result_df.dropna()
        result_df['ticker'] = ticker
        fin = pd.concat([fin, result_df], ignore_index=True)
        fin = fin.rename(columns={'fiscalDateEnding': 'Date'})
    
fin.to_csv(file_path0, index=False)
