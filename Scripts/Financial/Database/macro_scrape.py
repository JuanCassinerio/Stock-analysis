"""
macro variables RF / RM /
https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL + personal account api key => 1ad73d818bcf7a9313ed3bac1802f40b
!pip install pandas_datareader
!pip install fredapi
"""
#libraries
import datetime
from fredapi import Fred
import yfinance as yf
import warnings

from bs4 import BeautifulSoup

import zipfile
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO


import requests
import json
import pandas as pd
from datetime import date
import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.renderers.default = 'browser'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def salesprojection_exfall_rise(t,g, a, b):
    return g + np.exp(-a * (t - b))


def data_fitting(function,parameters_0,x,y,maxiters):

    #scaled data for big mangnitude difference between X and Y
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x.values.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    parameters, _ = curve_fit(function, x_scaled, y_scaled, p0=parameters_0, maxfev=maxiters)
    y_pred = function(x_scaled, *parameters)
    r2 = r2_score(y_scaled, y_pred)

    return paremeters,r2

def data_projection(function,parameters,x,y,m_projected):




    extrapolated_data = []
    x1 = df['Year'].iloc[-1] + 1

    for i in range(m + 1):
        next_x = x1 + i
        next_y = function(next_x, *parameters)
        extrapolated_data.append([next_x, next_y])
    ex = pd.DataFrame(extrapolated_data, columns=['x', 'y_RF'])


    for i in range(0,20)
        if i==20:
            error='max iters reached'
            exit()
        future_year = Datelast_date + pd.DateOffset(years=1)
        future_year_scaled = scaler_x.transform(future_years.year.values.reshape(-1, 1)).flatten()
        revenue_change = scaler_y.inverse_transform(best_function(future_year_scaled, *best_fit_params).reshape(-1, 1))  # unScale
        revenue *= revenue_change
        future_years.append(future_year)
        revenues.append(revenue)


    return ex

def fred():
    #folder_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/macro.csv'

    end_date = date.today()
    start_date = end_date - pd.DateOffset(years=80)
    api_key ='1ad73d818bcf7a9313ed3bac1802f40b'
    fred = Fred(api_key=api_key)

    #m = fred.get_series("WM2NS", start_date=start_date, end_date=end_date)  # Retrieve the M0 Money Stock data
    cpi = fred.get_series("CPIAUCSL", start_date=start_date, end_date=end_date)  # Retrieve the cpi
    rn=fred.get_series("FEDFUNDS",start_date=start_date, end_date=end_date)  #federal fund rate usa(overnight)
    rf= fred.get_series("DGS10", start_date=start_date, end_date=end_date) #Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis

    #df = pd.DataFrame({'cpi': cpi,'M0': m,'rf': rf,'rn': rn})
    df = pd.DataFrame({'rf': rf,'rn': rn,'cpi': cpi})
    df = df.resample('ME').mean()
    df['Date'] = df.index.strftime('%Y-%m')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    #df['Inflation_M'] = df['M0'].pct_change(12) * 100  # Calculate annual inflation rate
    df['Inflation'] = df['cpi'].pct_change(12,fill_method=None) * 100  # Calculate annual inflation rate

    df['rn.real'] = df['rn']-df['Inflation']
    df = df.dropna()

'''
    ticker='^GSPC'
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Marketreturn']=stock_data['Adj Close'].pct_change(252) * 100
    df = pd.merge(df, stock_data, on='Date')
    df = df.rename(columns={'Adj Close': 'SPY'})
    df = df[['Date', 'rf', 'Marketreturn','rn','rn.real']]
    df['SPY-rf'] = df['Marketreturn'] - df['rf']
    df.dropna()
    df['Mean_SPY-rf'] = df['SPY-rf'].expanding().mean()
    df['Mean_SPY-rf'] =df['SPY-rf'].rolling(window=12*30).mean()
    df['Year'] = df['Date'].dt.year'''




    dmd = dmd[['Year', 'Implied ERP (FCFE)']]
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
'''
    # data fitting
    
    
    print(r2)'''

    #data projection





    dffl = pd.concat([df,ex], ignore_index=True)

    df = df[['Date','RF','MP','Year']]
    df.to_csv(folder_path0, index=False)




    return fred







    '''
    
    
    # Plot

    fig = px.line(df, x=df.index, y='annual_inflation', title='Dolar Inflation U.S. (12 months)')  # Existing line
    fig.add_trace(go.Scatter(x=df.index, y=[inflation_mean] * len(df.index),
                             mode='lines',
                             name='Average Inflation',
                             line=dict(dash='dash')))  # Set line dash style to 'dash'
    fig.add_trace(go.Scatter(x=df.index, y=[inflation_mean - inflation_std] * len(df.index),
                             mode='lines',
                             name='Average Inflation',
                             line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=[inflation_mean + inflation_std] * len(df.index),
                             mode='lines',
                             name='Average Inflation',
                             line=dict(dash='dash')))
    fig.show()'''
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


def gdpworld():


    file_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/gdp growth.csv'

    url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=csv"

    content_stream = BytesIO(requests.get(url).content)
    z = zipfile.ZipFile(content_stream, 'r')
    csv_filename = z.namelist()[1]
    csv_file = z.open(csv_filename)
    # Skip the first 2 rows (header not present)
    header_rows = 4
    df = pd.read_csv(csv_file, encoding='latin1', skiprows=header_rows)
    df = df[df['Country Name'] == 'World']  # United States # World
    df = df.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])
    df = df.T
    df = df.dropna()
    df = df.rename(columns={259: 'gdp'})  # 259 for world #251 for US
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])

    def projection(x, a, b):
        return a * x + b

    df['Year'] = df['Date'].dt.year
    df = df[df['Year'] >= 1980][['Year','gdp']]
    df = df.rename(columns={'Year':'Date','gdp': 'gdp growth'})

    popt, _ = curve_fit(projection, df['Date'], df['gdp growth'], maxfev=100)  # p0=[5E+09, - 1E+13]
    ym = projection(df['Date'], *popt)

    x = df['Date']
    y = df['gdp growth']

    m = 20
    extrapolated_data = []
    x1 = df['Date'].iloc[-1] + 1
    for i in range(m + 1):
        next_x = x1 + i
        next_y = projection(next_x, *popt)
        extrapolated_data.append([next_x, next_y])
    ex = pd.DataFrame(extrapolated_data, columns=['x', 'y'])
    ex = ex.rename(columns={'x': 'Date', 'y': 'gdp growth'})

    g = pd.concat([df, ex], axis=0)


    '''
    ex.to_csv(file_path0, index=False)
    import plotly.io as pio
    import plotly.graph_objs as go
    pio.renderers.default='browser'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ym, mode='markers', name='fitting'))
    fig.add_trace(go.Scatter(x=ex['Date'], y=ex['gdp growth'], mode='markers', name='extrapolation data'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='present data'))
    fig.update_layout(width=700,height=400)
    fig.show()
    '''
    return gdp


def dmd(path):
    '''
    https: // pages.stern.nyu.edu / ~adamodar /
    https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html
    '''
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
    dmd = pd.DataFrame(data)
    dmd =dmd.dropna()['Implied ERP (FCFE)']
    '''ERP = pd.read_excel(path / 'histimpl.xls', engine='xlrd')
    first_non_nan_row = ERP.dropna().index[0]

    ERP.columns = ERP.iloc[first_non_nan_row]
    ERP = ERP.drop(first_non_nan_row, axis=0)[['Year','Implied ERP (FCFE)']]
    ERP = ERP.dropna()
    ERP['Date'] = pd.to_datetime(ERP['Year'])
    ERP['ERP'] = pd.to_numeric(ERP['Implied ERP (FCFE)'], errors='coerce')*100

    ERP = ERP[['Year', 'ERP']]
    '''

    return dmd


