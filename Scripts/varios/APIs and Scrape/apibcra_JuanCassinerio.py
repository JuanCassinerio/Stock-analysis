import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
# Disable SSL certificate verification
url = 'https://api.bcra.gob.ar/estadisticas/v1/principalesvariables'

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
data_json = requests.get(url, headers=headers, verify=False).json()
df = pd.DataFrame(data_json['results']) 
df = df[['idVariable','descripcion']]
print(df) 

id_variable=25 #1 es el id de la variable 

startdate="2010-01-01"
enddate=datetime.datetime.today().strftime('%Y-%m-%d')
'''
4=dolar oficial
27=inflacion mensual
25=m2
6=tasa
7=badlar
'''

url = f'https://api.bcra.gob.ar/estadisticas/v1/datosvariable/{id_variable}/{startdate}/{enddate}' 
data_json = requests.get(url, headers=headers, verify=False).json()
datahistorica = pd.DataFrame(data_json['results'])
description = df['descripcion'][df['idVariable'] == id_variable].iloc[0]
datahistorica['valor'] = datahistorica['valor'].str.replace('.', '').str.replace(',', '.').astype(float)
datahistorica['fecha'] = pd.to_datetime(datahistorica['fecha'], format='%d/%m/%Y')


plt.figure(figsize=(10, 6))  # Set plot size
plt.plot(datahistorica['fecha'], datahistorica['valor'])
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title(f"{description} - Valores Históricos")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

'''
vendedores de dolares=exportador/particular(el que no llega a fin de mes)/turista que entra/empresa extranjera que invierte/especulador
comprador= importador/particular(el que ahorra)/turista que sale/especulador



'''