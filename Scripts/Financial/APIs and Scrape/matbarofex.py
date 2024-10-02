import requests
import pandas as pd
from pathlib import Path


def integraciones_cotizaciones(instrument_codes, environment):
    if isinstance(instrument_codes, str):
        instrument_codes = [instrument_codes]

    request_body = {"instrumentos": []}

    for i in range(len(instrument_codes)):
        request_body["instrumentos"].append(
            {
                "codigoInstrumento": instrument_codes[i],
                "codigoMercado": "ROFX",
                "fechaHistoricoDesde": "2000-01-01"
            }
        )

    if environment.upper() == 'PRD':
        api_key = '9546a28eb54541f7bce51ca81826357a'
        url = ('https://apim-integraciones-prd-001.azure-api.net/mercados/'
               'cotizaciones')
    elif environment.upper() == 'STG':
        api_key = 'd8bbedcf4b5d4013a2ff5b8c344c9816'
        url = ('https://apim-integraciones-stg-001.azure-api.net/mercados/'
               'cotizaciones')
    else:
        raise ValueError('Invalid environment. Valid values are PRD and STG.')
    # key provided by Guido Palacios

    headers = {'Ocp-Apim-Subscription-Key': '{key}'.format(key=api_key)}
    api_response = requests.post(url, headers=headers,
                                 json=request_body).json()
    return api_response
