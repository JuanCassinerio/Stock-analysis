'''
ALPHAVANTAGE https://www.alphavantage.co  KEY 'B6T9Z1KKTBKA2I1C'
'''

import requests
import pandas as pd

def av_financials(ticker)

    key = 'B6T9Z1KKTBKA2I1C'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

    sections = ['BALANCE_SHEET', 'INCOME_STATEMENT', 'CASH_FLOW']

    ticker= 'AAPL'
    result_df = pd.DataFrame()
    for section in sections:
        url = f'https://www.alphavantage.co/query?function={section}&symbol={ticker}&apikey={key}'

        r = requests.get(url, headers=headers)
        if r.status_code == 200:  # Check if request is successful
            data = r.json()
            if len(data) == 1:
                break
            if "quarterlyReports" in data:  # Check if quarterlyReports exist in data
                quarterly_reports = data["quarterlyReports"]
                data = pd.DataFrame.from_dict(quarterly_reports)
                if result_df.empty:
                    result_df = data.copy()
                else:
                    result_df = pd.merge(result_df, data, on='fiscalDateEnding', how='left')
            else:
                print(f"Not complete quarterly reports for {ticker}")
                continue
        else:
            print(f"Fail request. Status code: {r.status_code}")

    if all_sections_data:
        result_df = result_df.dropna()
        result_df['ticker'] = ticker
    return result_df

