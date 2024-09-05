import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
# Disable SSL certificate verification
url = 'https://api.bcra.gob.ar/estadisticas/v2.0/PrincipalesVariables'

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
data_json = requests.get(url, headers=headers, verify=False).json()
df = pd.DataFrame(data_json['results']) 
df = df[['idVariable','descripcion']]
print(df) 

id_variable=15 #1 es el id de la variable

startdate="2024-01-01"
enddate=datetime.datetime.today().strftime('%Y-%m-%d')
'''
4=dolar oficial
27=inflacion mensual
25=m2
6=tasa
7=badlar
'''

url = f'https://api.bcra.gob.ar/estadisticas/v2.0/datosvariable/{id_variable}/{startdate}/{enddate}'
data_json = requests.get(url, headers=headers, verify=False).json()
datahistorica = pd.DataFrame(data_json['results'])
description = df['descripcion'][df['idVariable'] == id_variable].iloc[0]
#datahistorica['valor'] = datahistorica['valor'].str.replace('.', '').str.replace(',', '.').astype(float)
#datahistorica['fecha'] = pd.to_datetime(datahistorica['fecha'], format='%d/%m/%Y')

import matplotlib

import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg'

plt.figure(figsize=(10, 6))  # Set plot size
plt.plot(datahistorica['fecha'], datahistorica['valor'])
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title(f"{description} - Valores Históricos")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.grid(True)
plt.tight_layout()
plt.show()



# Sample plotting code
plt.figure(figsize=(10, 6))
plt.plot(df_quotes['Date'], df_quotes['SettlePrice'])

# Set the major locator to limit the number of ticks on the x-axis
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Adjust nbins to change tick frequency

# Alternatively, for time series data, you can format the x-axis with fewer ticks
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))

# Rotate dates for better readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.xlabel('Date')
plt.ylabel('Settle Price')
plt.title('Settle Price Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

'''
vendedores de dolares=exportador/particular(el que no llega a fin de mes)/turista que entra/empresa extranjera que invierte/especulador
comprador= importador/particular(el que ahorra)/turista que sale/especulador



'''