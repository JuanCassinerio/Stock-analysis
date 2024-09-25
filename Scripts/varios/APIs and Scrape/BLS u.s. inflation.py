"""
#BLS Bureau of Labor statistics / Inflation
https://data.bls.gov/dataViewer/view/timeseries/CUUR0000SA0L1E
https://www.bls.gov/developers/api_python.htm#python2
https://www.bls.gov/bls/api_features.htm
"""
import requests
import json
import pandas as pd
from datetime import datetime
import numpy as np

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.renderers.default='browser'
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



inflation_mean=df['annual_inflation'].mean()
inflation_std=df['annual_inflation'].std()

# Plot



fig = px.line(df, x=df.index, y='annual_inflation', title='Dolar Inflation U.S. (12 months)')  # Existing line
fig.add_trace(go.Scatter(x=df.index, y=[inflation_mean] * len(df.index),
                         mode='lines',
                         name='Average Inflation',
                         line=dict(dash='dash')))  # Set line dash style to 'dash'
fig.add_trace(go.Scatter(x=df.index, y=[inflation_mean-inflation_std] * len(df.index),
                         mode='lines',
                         name='Average Inflation',
                         line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=df.index, y=[inflation_mean+inflation_std] * len(df.index),
                         mode='lines',
                         name='Average Inflation',
                         line=dict(dash='dash')))
fig.show()

