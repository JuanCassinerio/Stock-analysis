################################################################
########################### BCRA API ###########################
################################################################
'''
https://www.bcra.gob.ar/Pdfs/PublicacionesEstadisticas/bolmetes.pdf
'''

import requests
import pandas as pd

import datetime

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


    import plotly.graph_objects as go
    import plotly.express as px

    # Create the figure
    fig = go.Figure()

    # Add the trace (line plot)
    fig.add_trace(go.Scatter(x=datahistorica['fecha'],
                             y=datahistorica['valor'],
                             mode='lines',
                             name=f'{description} - Valores Históricos'))

    # Update layout settings
    fig.update_layout(
        title=f"{description} - Valores Históricos",
        xaxis_title="Fecha",
        xaxis=dict(
            tickangle=0,
            nticks=10
        ),
        autosize=False,
        width=1000,
        height=600,
        template='plotly_white'
    )

    # Show the figure
    fig.show()

    return datahistorica

#PLOT and download
id_variable=15
datahistorica=data_plot(id_variable)

