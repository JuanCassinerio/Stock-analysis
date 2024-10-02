#libraries
import pandas as pd
import datetime
import yfinance as yf
import numpy as np
import warnings
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

file_path1 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'
file_path2 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/price.csv'



fin= pd.read_csv(file_path1)
price= pd.read_csv(file_path2)

fin['ticker','fiscalDateEnding','commonStockSharesOutstanding','netIncome_x','operatingCashflow','capitalExpenditures']
df = df.rename(columns={'netIncome_x': 'NetIncome', 'fiscalDateEnding': 'Date'})

into date
into floate

fin['fcf']=fin['operatingCashflow'] - fin['capitalExpenditures'] 
fin = fin[['ticker','Dateg','commonStockSharesOutstanding','NetIncome','fcf']]


df=merge(price,fin)


df['PE']=df['price']*df['commonStockSharesOutstanding']/df['NetIncome'] # MarketPrice/Earnings (NetIncome)
df['PF']=df['price']*df['commonStockSharesOutstanding']/df['fcf'] # MarketPrice/FCF

moving average
constant

'''
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=ym, mode='markers', name='fitting'))
fig.add_trace(go.Scatter(x=ex['Date'], y=ex['gdp growth'], mode='markers', name='extrapolation data'))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='present data'))
fig.update_layout(width=700,height=400)
fig.show()







def strategy_applier (df):
    
    if:
    elif:
    else:
        
    position
    return

#Strategy_score
def strategy_score (df):
    
    position value
    
    return

return and Sharpe(return for a given volatility)


vs buy and hold
vs sp500 buy and hold




'''

'''


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=ym, mode='markers', name='fitting'))
fig.add_trace(go.Scatter(x=ex['Date'], y=ex['gdp growth'], mode='markers', name='extrapolation data'))
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='present data'))
fig.update_layout(width=700,height=400)
fig.show()
'''