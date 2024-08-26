import pandas as pd
import requests
from io import BytesIO
import zipfile
from scipy.optimize import curve_fit

file_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/gdp growth.csv'

url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.KD.ZG?downloadformat=csv"

content_stream = BytesIO(requests.get(url).content)
z=zipfile.ZipFile(content_stream, 'r')
csv_filename = z.namelist()[1]
csv_file=z.open(csv_filename)
# Skip the first 2 rows (header not present)
header_rows = 4
df = pd.read_csv(csv_file, encoding='latin1', skiprows=header_rows)

df = df[df['Country Name'] == 'World']  # United States # World
df = df.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])
df = df.T

df=df.dropna()
df = df.rename(columns={259: 'gdp'}) #259 for world #251 for US
df['Date'] = df.index
df['Date'] = pd.to_datetime(df['Date'])

def projection(x,a,b):
    return a*x+b
df['Year'] = df['Date'].dt.year
df = df[df['Year'] >= 1980]

popt, _ = curve_fit(projection, df['Year'], df['gdp'], maxfev=100) #p0=[5E+09, - 1E+13]
ym = projection(df['Year'], *popt)

x=df['Year']
y=df['gdp']


m=15
extrapolated_data = []
x1 = df['Year'].iloc[-1] +1 
for i in range(m+1):
    next_x = x1 + i
    next_y = projection(next_x, *popt)
    extrapolated_data.append([next_x, next_y])
ex= pd.DataFrame(extrapolated_data, columns=['x', 'y'])
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
