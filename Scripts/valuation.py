#0 libraries and functions
import pandas as pd
from scipy.optimize import curve_fit

#from damodaranfcf import damodaranfcf

'''
#valuate one company example https://valueinvesting.io/MSFT/valuation/wacc
'''
#1 files
file_path1 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'
file_path2 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/macro.csv'
file_path3 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/price.csv'





file_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'

fin=pd.read_csv(file_path0)






fin= pd.read_csv(file_path1)
macro= pd.read_csv(file_path2)
price= pd.read_csv(file_path3)

tickers = fin['ticker'].unique().tolist()
#

#fin data preparation
ticker='MSFT'

fin = fin[fin['ticker'] == ticker]
fin['fiscalDateEnding'] = pd.to_datetime(fin['fiscalDateEnding'])
fin = fin.rename(columns={'fiscalDateEnding': 'Date','cashAndCashEquivalentsAtCarryingValue':'Cash','commonStockSharesOutstanding':'Shares'})
for col in ['Shares', 'totalNonCurrentLiabilities', 'Cash','operatingCashflow','capitalExpenditures','incomeTaxExpense','incomeBeforeTax','interestExpense']:
    fin[col] = pd.to_numeric(fin[col], errors='coerce')  # Convert to numeric, set errors to NaT
fin['fcf']=fin['operatingCashflow'] - fin['capitalExpenditures'] 
fin['tax'] = fin['incomeTaxExpense'] / fin['incomeBeforeTax']
reversed_tax_ratio = fin['tax'].iloc[::-1]
fin['tax'] = reversed_tax_ratio.rolling(window=16, min_periods=16).median().iloc[::-1] #mean
reversed_interest_expense = fin['interestExpense'].iloc[::-1]
fin['interestExpensecumm'] = reversed_interest_expense.rolling(window=4, min_periods=4).sum().iloc[::-1]
fin['kd']=fin['interestExpensecumm']/fin['totalNonCurrentLiabilities']
fin = fin[['Date','fcf','Shares','totalNonCurrentLiabilities','Cash', 'interestExpense','interestExpensecumm','tax','kd','totalNonCurrentLiabilities']]
fin = fin[['Date', 'fcf','Shares','totalNonCurrentLiabilities','Cash','kd','tax']]


#
# project 

#def salesprojection(x,a1,b1,c1,g,b2,c2):
#    return a1*(1+b1/100)**(x-c1)+g/(1+np.exp(b2*(x-c2)))
















#macro

my=20 #years
macrof= df[df['Year'] >= macro['Year'].max()-my]

def curve(x,a,b):
    return a*x+b

def projection(x):
    
    popt, _ = curve_fit(curve, macrof['Year'], macrof[x], maxfev=100) 
    
    
    return projection(macrof['Year'], *popt)

y_RF = projection('RF')

popt_MP, _ = curve_fit(projection, dff['Year'], dff['MP'], maxfev=100) 
y_MP = projection(dff['Year'], *popt_MP)

m=15
extrapolated_data = []
x1 = df['Year'].iloc[-1] +1 
for i in range(m+1):
    next_x = x1 + i
    next_y = projection(next_x, *popt_RF)
    extrapolated_data.append([next_x, next_y])
ex= pd.DataFrame(extrapolated_data, columns=['x', 'y_RF'])
ex_RF = ex.rename(columns={'x': 'Date'})

m=15
extrapolated_data = []
x1 = df['Year'].iloc[-1] +1 
for i in range(m+1):
    next_x = x1 + i
    next_y = projection(next_x, *popt_MP)
    extrapolated_data.append([next_x, next_y])
ex= pd.DataFrame(extrapolated_data, columns=['x', 'y_MP'])
ex_MP = ex.rename(columns={'x': 'Date'})


ex = pd.merge(ex_RF , ex_MP, on='Date')
ex = ex.rename(columns={'Date': 'Year','y_RF': 'RF','y_MP': 'MP'})




#calculate ke to each new stock date
tickers = fin['ticker'].unique().tolist() 
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

    





get years to statioanrity



























fin.to_csv(file_path0, index=False)