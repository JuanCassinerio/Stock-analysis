from pathlib import Path

import pandas as pd

from ticker_list import comafi

if user=__main__


    cwd = Path.cwd()
    output_path = cwd /'db'

    price=pd.DataFrame(output_path/'price.csv')
    financials_statements=pd.DataFrame(output_path/'financials_statements.csv')
    companydescription=pd.DataFrame(output_path/'companydescription.csv')
df = pd.read_excel(output_path/'histimpl.xls', engine='openpyxl')
    ERP=pd.DataFrame(output_path/'histimpl.xls')
    macro=pd.DataFrame(output_path/'macro.csv')

    #execute all data scrapers
    cedear_tickerlist=comafi()

    if len(cedear_tickerlist) != 0:
        ticker_list=cedear_tickerlist



    from ticker_scrape import price,financials,companydescription
    for ticker in tickerlist
        price
        financials
        companydescription

    #macro





    #append data

    #save data and upload new data folder to cloud + file with date of dataupload
    ticker_list.to_csv(output_path / 'ticker_list')
    financial_statements.to_csv( output_path/'financials.csv')
    price.to_csv( output_path/'financials.csv')
    financial_statements.to_csv( output_path/'financials.csv')
    financial_statements.to_csv( output_path/'financials.csv')


