"""
macro variables RF / RM /
https://fred.stlouisfed.org/series/CORESTICKM159SFRBATL + personal account api key => 1ad73d818bcf7a9313ed3bac1802f40b
!pip install pandas_datareader
!pip install fredapi
"""
#libraries
import datetime
from fredapi import Fred
import warnings
from bs4 import BeautifulSoup
import zipfile
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from functools import partial
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


def data_fitting(function,x,y):
    x=df_fitting['Date']
    y=df_fitting['rf']
    function=salesprojection_exfall_rise
    maxiters=100000

    #scaled data for big mangnitude difference between X and Y
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x.values.reshape(-1, 1)).flatten()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    parameters, _ = curve_fit(function, x_scaled, y_scaled, maxfev=maxiters)
    y_pred = function(x_scaled, *parameters)
    r2 = r2_score(y_scaled, y_pred)

    return parameters,r2

def data_projection(function,parameters,x,y,m_projected):
    extrapolated_data = []
    x0 = x.iloc[-1] + pd.DateOffset(years=1)
    m_projected=10
    for i in range(m_projected + 1):
        next_x = x0 + + pd.DateOffset(years=1)*i
        next_x_scaled = scaler_x.transform(future_years.year.values.reshape(-1, 1)).flatten()
        next_y_scaled = function(next_x_scaled, *parameters)
        next_y = scaler_y.inverse_transform(next_y_scaled)  # unScale
        extrapolated_data.append([next_x, next_y])
    ex = pd.DataFrame(extrapolated_data, columns=['x', 'y'])

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
    df = df[df['Date'] >= '2008-01-01 00:00:00'][['Date','rf']]

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
    df['Year'] = df['Date'].dt.year

    df['rf'] = df['rf'].interpolate(method='linear').round(2)



    # data fitting
    df = df[df['Date'] >= '2008-01-01 00:00:00'][['Date','rf']]

  
    fig = px.line(df_fitting, x='Date', y='rf', title='Dolar Inflation U.S. (12 months)')  # Existing line
    
    fig.show()
    
    fig = px.line( x=x_scaled, y=y_scaled, title='Dolar Inflation U.S. (12 months)')  # Existing line
    
  

    df_fitting = df[df['Date'] >= '2020-01-01 00:00:00'][['Date','rf']]


    fixed_g = df['rf'].mean()
    salesprojection_exfall_rise_partial = partial(salesprojection_exfall_rise, g=fixed_g)
    parameters,r2=data_fitting(salesprojection_exfall_rise_partial,df_fitting['Date'],df_fitting['rf'])

    ex=data_projection(function,parameters,_fitting,y,m_projected)


    print(r2)

    #data projection

    ex_rf=data_projection(function,parameters,x,y,m_projected)





    dffl = pd.concat([df,ex], ignore_index=True).reset.index
    df.to_csv(folder_path0, index=False)


        |'''

    return df







    '''
    
    
    # Plot

    fig = px.line(df, x='Date', y='rf', title='Dolar Inflation U.S. (12 months)')  # Existing line
    
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
    fig.show()
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
    return g

def dmd():
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
    dmd.columns = dmd.iloc[0]  # Set the first row as the header
    dmd = dmd[1:].reset_index(drop=True)  # Drop the first row and reset the index

    dmd['Implied ERP (FCFE)'] = dmd['Implied ERP (FCFE)'].replace(['', ' ', None], np.nan)

    # Step 2: Drop rows where 'Implied ERP (FCFE)' is NaN
    dmd = dmd.dropna(subset=['Implied ERP (FCFE)']).reset_index(drop=True)
    dmd['Implied ERP (FCFE)'] = dmd['Implied ERP (FCFE)'].str.replace('%', '', regex=False).astype(float)


    '''
    ERP = pd.read_excel(path / 'histimpl.xls', engine='xlrd')
    first_non_nan_row = ERP.dropna().index[0]

    ERP.columns = ERP.iloc[first_non_nan_row]
    ERP = ERP.drop(first_non_nan_row, axis=0)[['Year','Implied ERP (FCFE)']]
    ERP = ERP.dropna()
    ERP['Date'] = pd.to_datetime(ERP['Year'])
    ERP['ERP'] = pd.to_numeric(ERP['Implied ERP (FCFE)'], errors='coerce')*100

    ERP = ERP[['Year', 'ERP']]
    '''

    return dmd


