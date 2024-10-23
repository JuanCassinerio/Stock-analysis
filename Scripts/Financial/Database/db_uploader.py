from pathlib import Path
import pandas as pd
from ticker_list import comafi

from ticker_scrape import price,yf_financials,companydescription
from SEC EDGAR import secfsdstools

if __name__ == '__main__':
    '''
    'openpyxl' for xlsx
    xlrd for xls
    
    '''


    cwd = Path.cwd()
    path = cwd.parent.parent /'Database'/'db'

    price=pd.read_csv(path/'price.csv')
    financials_statements=pd.read_csv(path/'financials_statements.csv')
    companydescription=pd.read_csv(path/'companydescription.csv')
    #damodaran Implied ERP
    ERP = pd.read_excel(path/'histimpl.xls', engine='xlrd')
    ERP = ERP.dropna(thresh=ERP.shape[1] - 4)
    ERP.columns = ERP.iloc[0]
    ERP = ERP[1:]




    ERP=pd.DataFrame(path/'histimpl.xls')
    macro=pd.read_csv(path/'macro.csv')

    #execute all data scrapers
    cedear_tickerlist=comafi()

    if len(cedear_tickerlist) != 0:
        ticker_list=cedear_tickerlist



    from ticker_scrape import price,financials,companydescription

    secfsdstools()

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

########################


