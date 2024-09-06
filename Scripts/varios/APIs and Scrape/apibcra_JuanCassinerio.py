import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg'

# https://www.bcra.gob.ar/BCRAyVos/catalogo-de-APIs-banco-central.asp
url = 'https://api.bcra.gob.ar/estadisticas/v2.0/PrincipalesVariables'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
data_json = requests.get(url, headers=headers, verify=False).json()
df = pd.DataFrame(data_json['results']) 
df = df[['idVariable','descripcion']]
print(df)

#Catalogo de variables
'''
15=base monetaria
4=dolar oficial
27=inflacion mensual
25=m2
6=tasa
7=badlar
'''

def data_plot (id_variable):
    startdate = "2024-01-01"
    enddate = datetime.datetime.today().strftime('%Y-%m-%d')
    url = f'https://api.bcra.gob.ar/estadisticas/v2.0/datosvariable/{id_variable}/{startdate}/{enddate}'
    data_json = requests.get(url, headers=headers, verify=False).json()
    datahistorica = pd.DataFrame(data_json['results'])
    description = df['descripcion'][df['idVariable'] == id_variable].iloc[0]
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
    return datahistorica

#PLOT and download
id_variable=15
datahistorica=data_plot(id_variable)

