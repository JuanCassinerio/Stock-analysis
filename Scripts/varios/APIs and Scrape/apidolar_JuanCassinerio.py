#https://dolarapi.com/ = https://argentinadatos.com/

#for spyder-anaconda
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
from datetime import date
import requests 
import pandas as pd

url="https://api.argentinadatos.com/v1/cotizaciones/dolares"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
data = requests.get(url, headers=headers, verify=False).json()
data = pd.DataFrame.from_dict(data)
data['precio'] = (data['compra']+data['venta'])/2
data = data[['fecha','precio','casa']]
casa_values = data['casa'].unique()  # Get unique casa values
for casa in casa_values:
  data[f'{casa}'] = data[data['casa'] == casa]['precio']
data.drop(['casa','precio','solidario'], axis=1, inplace=True)
def fill_missing_by_fecha(df):
  return pd.concat([df[['fecha']], df.groupby('fecha').transform(lambda x: x.fillna(method='ffill'))], axis=1)
data1 = fill_missing_by_fecha(data.copy())  # Avoid modifying original data
data1.sort_index(ascending=False, inplace=True)  # Sort DataFrame by index in descending order
data1.drop_duplicates(subset='fecha', keep='first', inplace=True)  # Keep the first occurrence of each unique 'fecha'
data1.sort_index(ascending=True, inplace=True)  # Sort DataFrame by index in ascending order again


url="https://dolarapi.com/v1/dolares"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
data = requests.get(url, headers=headers, verify=False).json()
data = pd.DataFrame.from_dict(data)
data['precio'] = (data['compra']+data['venta'])/2
data = data[['fechaActualizacion','precio','casa']]
casa_values = data['casa'].unique()  # Get unique casa values
for casa in casa_values:
  data[f'{casa}'] = data[data['casa'] == casa]['precio']
data.drop(['casa','precio'], axis=1, inplace=True)
def fill_missing_by_fecha(df):
  return pd.concat([df[['fechaActualizacion']], df.groupby('fechaActualizacion').transform(lambda x: x.fillna(method='ffill'))], axis=1)
data2 = fill_missing_by_fecha(data.copy())  # Avoid modifying original data
data2.sort_index(ascending=False, inplace=True)  # Sort DataFrame by index in descending order
data2.drop_duplicates(subset='fechaActualizacion', keep='first', inplace=True)  # Keep the first occurrence of each unique 'fecha'
data2.sort_index(ascending=True, inplace=True)  # Sort DataFrame by index in ascending order again
data2['fechaActualizacion'] = pd.to_datetime(data['fechaActualizacion'])  # Convert to datetime if not already
data2['fechaActualizacion'] -= pd.Timedelta(hours=1)

data2 = data2.rename(columns={'fechaActualizacion': 'fecha'})
data2['fecha'] = pd.to_datetime(data2['fecha']).dt.strftime('%Y-%m-%d')


data = pd.concat([data1, data2], ignore_index=True)
data['fecha'] = pd.to_datetime(data['fecha'])

# Convert 'fecha' column to datetime


data.drop_duplicates
# Get today's date and extract the date part
today = date.today()
start_date = "2024-01-01"
start_date_date = pd.to_datetime(start_date).date()

filtered_df = data[data['fecha'].dt.date >= start_date_date]
filtered_df = filtered_df[filtered_df['fecha'].dt.date <= today]


# Plot
fig = px.line(filtered_df, x='fecha', y='contadoconliqui', title='Dolar CCL')
fig.show()

'''
# Plot
fig = px.line(filtered_df, x='fecha', y='contadoconliqui')

# Make the title clickable
fig.update_layout(
    title=dict(
        text='Dolar CCL - <a href="https://juancassinerio.wixsite.com/finance">juancassinerio.wixsite.com/finance</a>',
        x=0.5,
        xanchor='center',
        font=dict(color="blue", size=14)
    )
)

fig.show()


'''


####
'''
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

n = 2# years

ccl = pd.read_csv('DOLAR CCL - Cotizaciones historicas.csv')
ccl['fecha'] = pd.to_datetime(ccl['fecha'], format='%Y-%m-%d')
ccl.set_index('fecha', inplace=True)
ccl = ccl[ccl.index >= ccl.index.max() - pd.DateOffset(years=n)]
ccl = pd.DataFrame({'Date': ccl.index, 'ccl': ccl['cierre']})
ccl.reset_index(drop=True, inplace=True)  # Resetting the index

def func(x, a, b, c):
    return a*((b)**(x-c) +1)
popt, _ = curve_fit(func, ccl.index, ccl['ccl'], maxfev=1000)
trend_ccl = func(ccl.index, *popt)

fig = make_subplots(rows=2, cols=1, subplot_titles='')
fig.add_trace(go.Scatter(x=ccl['Date'], y=ccl['ccl'], name='Dollar CCL Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=trend_ccl, name='Trend'), row=1, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=(ccl['ccl'] - trend_ccl)*100/trend_ccl, name='Cycle + error(%)', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=[0] * len(ccl.index), mode='lines', name='Zero Line', line=dict(color='gray', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=[10] * len(ccl.index), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=[-10] * len(ccl.index), mode='lines', name='Zero Line', line=dict(color='green', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=[5] * len(ccl.index), mode='lines', name='Zero Line', line=dict(color='orange', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=ccl['Date'], y=[-5] * len(ccl.index), mode='lines', name='Zero Line', line=dict(color='light green', dash='dash')), row=2, col=1)
fig.update_layout(xaxis_title='Date', yaxis_title='Price (ARS)', height=900, width=1000)
fig.show()

d=1200 #current dolar

import plotly.express as px
fig = px.histogram((ccl['ccl'] - trend_ccl)*100/trend_ccl, nbins=1000, labels={'value': 'Variable Value', 'count': 'Frequency'}, title='Histogram of the Variable')
fig.update_layout(height=400, width=1500)
fig.show()

m=((ccl['ccl'] - trend_ccl)*100/trend_ccl).mean()
dv=((ccl['ccl'] - trend_ccl)*100/trend_ccl).std()
ddv=((ccl['ccl'] - trend_ccl)*100/trend_ccl).std()*4
cd=((ccl['ccl'] - trend_ccl)*100/trend_ccl).iloc[-1]

p = ((cd - m) / m) * 100
if p < 0:
  status = 'Comprar'
else:
  status = 'Vender'
dolar = []
dolar.append({'volatility 63%': dv, 'Volatility 99.5% Percentage Difference': ddv,'current desv': cd, 'Status': status})
dolar_df = pd.DataFrame(dolar)
dolar_df
'''


