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
from scipy.optimize import curve_fit


from io import BytesIO


import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

pio.renderers.default = 'browser'


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fred():
#folder_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/macro.csv'

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




def inflation():
    """
    #BLS Bureau of Labor statistics / Inflation
    https://data.bls.gov/dataViewer/view/timeseries/CUUR0000SA0L1E
    https://www.bls.gov/developers/api_python.htm#python2
    https://www.bls.gov/bls/api_features.htm
    """
    def datagather(data):
        # Send the request to the BLS API
        json_data = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers).json()

        # Parse the data and prepare for DataFrame
        data_list = []
        for series in json_data['Results']['series']:
            seriesId = series['seriesID']
            for item in series['data']:
                data_list.append({
                'Series ID': seriesId,
                'Year': int(item['year']),
                'Month': item['period'],
                'value': float(item['value'])
                })

        return  pd.DataFrame(data_list)

    headers = {'Content-type': 'application/json'}

    # Get the current year
    current_year = datetime.now().year
    years_back = 20  # How many years back from the current year

    # Generate data packages in 10-year intervals
    data_intervals = []
    for start in range(current_year - years_back, current_year, 10):
        end = min(start + 10, current_year)  # Ensure the last interval doesn't exceed the current year
        data = json.dumps({
            "seriesid": ['CUSR0000SA0'],
            "startyear": str(start),
            "endyear": str(end)
        })
        data_intervals.append(data)

    # Gather the data for each package and store it in a DataFrame
    dfs = []
    for data in data_intervals:
        df = datagather(data)
        dfs.append(df)

    # Concatenate all the dataframes
    df = pd.concat(dfs, ignore_index=True)



    df['Date'] = df['Year'].astype(str) + df['Month']
    df.index = df['Date'].apply(lambda x: datetime.strptime(x, '%YM%m'))

    df.sort_index(inplace=True)

    df['cpi'] = df['value'].pct_change(periods=1) #monthly

    df['annual_inflation'] = ((1 + df['cpi']).rolling(window=13).apply(lambda x: np.prod(x)) - 1)*100



    '''
    inflation_mean = df['annual_inflation'].mean()
    inflation_std = df['annual_inflation'].std()
    
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

    return df

def gdpworld():
    import pandas as pd
    import requests
    from io import BytesIO
    import zipfile
    from scipy.optimize import curve_fit

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
    df = df[df['Year'] >= 1980]

    popt, _ = curve_fit(projection, df['Year'], df['gdp'], maxfev=100)  # p0=[5E+09, - 1E+13]
    ym = projection(df['Year'], *popt)

    x = df['Year']
    y = df['gdp']

    m = 15
    extrapolated_data = []
    x1 = df['Year'].iloc[-1] + 1
    for i in range(m + 1):
        next_x = x1 + i
        next_y = projection(next_x, *popt)
        extrapolated_data.append([next_x, next_y])
    ex = pd.DataFrame(extrapolated_data, columns=['x', 'y'])
    ex = ex.rename(columns={'x': 'Date', 'y': 'gdp growth'})

    ex.to_csv(file_path0, index=False)

    '''
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
    return ex


def MarketPremium():
    '''
    https: // pages.stern.nyu.edu / ~adamodar /
https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html
    '''

    url = "https://www.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls"

    df=[]
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for non-200 status codes

        # Download data in chunks
        data = BytesIO()
        for chunk in response.iter_content(1024):
            data.write(chunk)

        # Load data as DataFrame (assuming XLS format)
        df = pd.read_excel(data, engine='xlrd')  # Use xlrd for .xls files

        print("Data successfully downloaded and converted to DataFrame.")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
    return df

