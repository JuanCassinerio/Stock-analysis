"""
macro variables RF / RM /
https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL + personal account api key => 1ad73d818bcf7a9313ed3bac1802f40b
!pip install pandas_datareader
!pip install fredapi
"""
#libraries
import pandas as pd
import datetime
from fredapi import Fred
import yfinance as yf
import warnings
import requests
from bs4 import BeautifulSoup
from scipy.optimize import curve_fit

import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


folder_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/macro.csv'

start_date = datetime.datetime.now() - datetime.timedelta(days=365*80)
end_date = datetime.datetime.now()

api_key ='1ad73d818bcf7a9313ed3bac1802f40b'
fred = Fred(api_key=api_key)

#m = fred.get_series("WM2NS", start_date=start_date, end_date=end_date)  # Retrieve the M0 Money Stock data
cpi = fred.get_series("CPIAUCSL", start_date=start_date, end_date=end_date)  # Retrieve the cpi
rn=fred.get_series("FEDFUNDS",start_date=start_date, end_date=end_date)  #federal fund rate usa(night
rf= fred.get_series("DGS10", start_date=start_date, end_date=end_date) #Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis

#df = pd.DataFrame({'cpi': cpi,'M0': m,'rf': rf,'rn': rn})
df = pd.DataFrame({'rf': rf,'rn': rn,'cpi': cpi})
df = df.resample('M').mean()
df['Date'] = df.index.strftime('%Y-%m')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
#df['Inflation_M'] = df['M0'].pct_change(12) * 100  # Calculate annual inflation rate
df['Inflation'] = df['cpi'].pct_change(12) * 100  # Calculate annual inflation rate

df['rn.real'] = df['rn']-df['Inflation']
df = df.dropna()


ticker='^GSPC'
stock_data = yf.download(ticker, start=start_date, end=end_date)
stock_data['Marketreturn']=stock_data['Adj Close'].pct_change(252) * 100
df = pd.merge(df, stock_data, on='Date')
df = df.rename(columns={'Adj Close': 'SPY'})
df = df[['Date', 'rf', 'Marketreturn','rn','rn.real']]
df['SPY-rf'] = df['Marketreturn'] - df['rf']
df.dropna()
df['Mean_SPY-rf'] = df['SPY-rf'].expanding().mean()
df['Year'] = df['Date'].dt.year

'''
start_date = datetime.datetime.now() - datetime.timedelta(days=365*60)
end_date = datetime.datetime.now()

ax = df.plot(x='Date', y='rn',  label='Rate daily')
df.plot(x='Date', y='rf', ax=ax, label='Rate 10 Bond')
df.plot(x='Date', y='Marketreturn', ax=ax, label='Marketreturn')

df.plot(x='Date', y='rn.real', ax=ax, label='rn.real')
ax.axhline(y=0, color='r', linestyle='--')
plt.legend()  # Add legend
plt.xlim(start_date,end_date)
plt.ylim(-20,50)
plt.show()
'''

# damodaran risk free rate and market premium(yearly)
url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html"
response = requests.get(url, verify=False)
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")
data = []
for row in table.find_all("tr"):
    row_data = []
    for cell in row.find_all(["th", "td"]):
        row_data.append(cell.get_text(strip=True))
    data.append(row_data)

columns = data[0]
dmd = pd.DataFrame(data[1:], columns=columns)

df['Year'] = df['Year'].astype(str)
df = pd.merge(df, dmd, on='Year', how='left')
df = df[['Date', 'rf', 'Marketreturn', 'Implied ERP (FCFE)']]
df['Date'] = df['Date'].dt.to_period('M')


start_date = df['Date'].min()
end_date = df['Date'].max()
date_range = pd.period_range(start=start_date, end=end_date, freq='M')
date_df = pd.DataFrame({'Date': date_range})
df = pd.merge(date_df, df, on='Date', how='left')

df['rf'] = pd.to_numeric(df['rf'], errors='coerce')
df['Marketreturn'] = pd.to_numeric(df['Marketreturn'], errors='coerce')
df['Implied ERP (FCFE)'] = df['Implied ERP (FCFE)'].str.rstrip('%').astype(float)

# Fill in missing values using the mean of the previous and next available values
df['rf'] = df['rf'].interpolate(method='linear').round(2)
df['Marketreturn'] = df['Marketreturn'].interpolate(method='linear').round(2)
df['Implied ERP (FCFE)'] = df['Implied ERP (FCFE)'].interpolate(method='linear').round(2)
#df['SPY'] = df['SPY'].interpolate(method='linear').round(2)

df = df.rename(columns={'rf':'RF'})
df = df.rename(columns={'Implied ERP (FCFE)': 'MP'})
df = df.rename(columns={'Marketreturn': 'RM'})


df['Year'] = df['Date'].dt.year
dff = df[df['Year'] >= 2008]

def projection(x,a,b):
    return a*x+b




popt_RF, _ = curve_fit(projection, dff['Year'], dff['RF'], maxfev=100) 
y_RF = projection(dff['Year'], *popt_RF)

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


'''
x=dff['Year']
y=dff['RF']

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_RF, mode='markers', name='fitting'))
fig.add_trace(go.Scatter(x=ex_RF['Date'], y=ex_RF['y_RF'], mode='markers', name='extrapolation data'))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='present data'))
fig.update_layout(width=700,height=400)
fig.show()
'''

'''
x=dff['Year']
y=dff['MP']

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_MP, mode='markers', name='fitting'))
fig.add_trace(go.Scatter(x=ex_MP['Date'], y=ex_MP['y_MP'], mode='markers', name='extrapolation data'))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='present data'))
fig.update_layout(width=700,height=400)
fig.show()
'''


dffl = pd.concat([df,ex], ignore_index=True)

df = df[['Date','RF','MP','Year']]
df.to_csv(folder_path0, index=False)








