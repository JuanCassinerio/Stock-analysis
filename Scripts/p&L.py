#libraries
import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#functions
from damodaranfcf import damodaranfcf

file_path1 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'
file_path2 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/macro.csv'
file_path3 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/price.csv'

'''
# Filterring stocks that have cedears
tickers = ['ABT', 'HWM', 'AEG', 'MO', 'AOCA', 'AEM', 'AMX', 'AXP', 'AIG', 'AMGN','ADI', 'AAPL', 'AMAT', 'T', 'ADP', 'AVY', 'CAR', 'NTCO', 'AZN', 'BBVA','BBD', 'BSBR', 'BAC', 'BCS', 'GOLD', 'BAS GR', 'BAYN GR', 'BHP', 'BA','BP', 'BMY', 'BG', 'CAH', 'CAT', 'CX', 'CVX', 'CSCO', 'C', 'KO', 'KOF','CDE', 'CL', 'VALE', 'GLW', 'COST', 'MBG GR', 'DE', 'DTEA GR', 'DEO','DIS', 'DD', 'EBAY', 'EOAN GR', 'LLY', 'E', 'XOM', 'FNMA', 'FDX', 'FSLR','FMX', 'ORAN', 'FMCC', 'FCX', 'GE', 'GSK', 'GFI', 'GOOGL', 'PAC', 'ASR','TV', 'BSN GR', 'HOG', 'HMY', 'HDB', 'HL', 'HPQ', 'HMC', 'HHPD LI', 'HON','HSBC', 'HNPIY', 'IBM', 'IBN', 'INFY', 'ING', 'INTC', 'IFF', 'IP', 'JPM','JNJ', 'KMB', 'KGC', 'KB', 'KEP', 'LVS', 'LYG', 'ERIC', 'LMT', 'MMC','PCRFY', 'MCD', 'MDT', 'MELI', 'MRK', 'MSFT', 'MMM', 'MUFG', 'MFG', 'MBT','MSI', 'NGG', 'NEC1 GR', 'NEM', 'NKE', 'NSANY', 'NOK', 'NMR', 'NVS', 'NUE','OGZD', 'LKOD', 'ATAD', 'NLMK LI', 'ORCL', 'PCAR', 'PSO', 'PEP', 'PTRCY','PFE', 'PHG', 'PBI', 'PKX', 'PG', 'QCOM', 'BB', 'RIO', 'SHEL', 'SMSN LI','SAP', 'SLB', 'SIEGY', 'SNA', 'SONY', 'SCCO', 'SBUX', 'SYY', 'TSM', 'TTM','TIIAY', 'TEF', 'TX', 'TXN', 'BK', 'HSY', 'HD', 'TTE', 'TM', 'TRV', 'JCI','USB', 'UL', 'X', 'RTX', 'VZ', 'VOD', 'WMT', 'WFC', 'AABA', 'YZCAY', 'META','AMZN', 'NVDA', 'ADBE', 'BIIB', 'GILD', 'NFLX', 'TMO', 'PYPL', 'CRM', 'TSLA','AMD', 'NG', 'TRIP', 'ANF', 'URBN', 'GRMN', 'SNAP', 'VRSN', 'XRX', 'YELP','ROST', 'TGT', 'GS', 'V', 'ARCO', 'DESP', 'AGRO', 'GLOB', 'BABA', 'BIDU','ABEV', 'VIV', 'JD', 'NTES', 'TCOM', 'YY', 'GGB', 'BRFS', 'CBD', 'SBS','WB', 'ITUB', 'ERJ', 'UGP', 'SUZ', 'EBR', 'ELP', 'TIMB', 'SID', 'TS', 'SAN','PBR', 'VIST', 'ABBV', 'BRKB', 'BIOX', 'AVGO', 'CAAP', 'DOCU', 'ETSY', 'GPRK','HAL', 'MA', 'PAAS', 'PSX', 'UNP', 'UNH', 'WBA', 'ZM', 'EFX', 'SQ', 'SHOP','SPOT', 'SNOW', 'TWLO', 'COIN', 'SPGI', 'AAL', 'LRCX', 'EA', 'XP', 'GM','DOW', 'AKO.B', 'NIO', 'SE', 'ADS']
print("S&P500 Tickers con CEDEAR:",len(tickers))
'''

fin= pd.read_csv(file_path1)
macro= pd.read_csv(file_path2)
price= pd.read_csv(file_path3)

tickers = fin['ticker'].unique().tolist() 
fin['fiscalDateEnding'] = pd.to_datetime(fin['fiscalDateEnding'])
df = df.rename(columns={'fiscalDateEnding': 'Date'})
for col in ['commonStockSharesOutstanding', 'totalNonCurrentLiabilities', 'cashAndCashEquivalentsAtCarryingValue','operatingCashflow','capitalExpenditures','incomeTaxExpense','incomeBeforeTax','interestExpense']:
    fin[col] = pd.to_numeric(fin[col], errors='coerce')  # Convert to numeric, set errors to NaT

fin['fcf']=fin['operatingCashflow'] - fin['capitalExpenditures'] 
fin['tax']=fin['incomeTaxExpense']/fin['incomeBeforeTax']
df = df[['Date', 'fcf','commonStockSharesOutstanding','totalNonCurrentLiabilities','cashAndCashEquivalentsAtCarryingValue','interestExpense','tax']]



#calculate ke to each new stock date
for ticker in tickers:
    ticker='MSFT'
    fin_s = fin[fin['ticker'] == ticker]
    macro_s = fin[fin['ticker'] == ticker]
    price_s = fin[fin['ticker'] == ticker]

    price['Month'] = price['Date'].dt.month
    price['Year'] = price['Date'].dt.year
    
    merge(macro,price)

    fin_s['interestExpense']

    fin_s['interestExpense'].rolling(window=4).sum().reset_index(drop=True)

    fin_s['kd'] = fin_s['interestExpense'].rolling(window=4).sum().reset_index(drop=True)/fin_s['totalNonCurrentLiabilities']


    fin_s['tarjet price'] = damodaran( date, fcf/ {g} {wacc} {t} /commonStockSharesOutstanding totalNonCurrentLiabilities cashAndCashEquivalentsAtCarryingValue)

    
    











#strategy aplier

def strategy:

    
    if ...
    return

    buy hold or sell
    if
    else if
    else

for ticker in tickers

    

#strategy results

def strategy:

    
    if ...
    return

    buy hold or sell
    if
    else if
    else

for ticker in tickers











#strategy and results for ticker append to dataframe column for ticker(return)

#mean result for tickers(weighted by marketcap to date), mean pot and individual plot(different colors dashed)






'''
https://rfachrizal.medium.com/how-to-obtain-financial-statements-from-stocks-using-yfinance-87c432b803b8

* https://www.geeksforgeeks.org/get-financial-data-from-yahoo-finance-with-python/

* https://github.com/ranaroussi/yfinance#readme

* https://pypi.org/project/yfinance/

* https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm

* https://www.gurufocus.com/stock/AAHTF/summary?search=AAPICO

* https://valueinvesting.io/AAPL/valuation/wacc

* https://ycharts.com/indicators/10_year_treasury_rate

* https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?end=2022&start=2013
'''
















