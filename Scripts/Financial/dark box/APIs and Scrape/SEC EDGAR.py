"""
https://www.youtube.com/watch?v=Wr1NoM3JkTo&t=3s&ab_channel=AdamGetbags
https://www.sec.gov/edgar/sec-api-documentation
https://www.sec.gov/os/webmaster-faq#developers

simplified libraries with standarized variables in 10_Q/K dataframes
sec_api
oder
pip install secfsdstools

https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
doenloads zip files of all financielstatements
turns them into database file and erase zip files


not all companies re publi traded, cik is a more generic code


It downloads all available quarterly zip files from https://www.sec.gov/dera/data/financial-statement-data-sets.html
    into the folder C:\Users\x1juacas\secfsdstools/data/dld
 2. It will transform the CSV files inside the zipfiles into parquet format and
    store them under C:\Users\x1juacas\secfsdstools/data/parquet
 3. The original zip file will be deleted depending on your configuration
 4. The content of the SUB.TXT parquet files will be indexed in a simple
    sqlite database file (placed at C:\Users\x1juacas\secfsdstools/data/db)


XBRL:Extensible Business Reporting Language
normalizar el formato con el que la información financiera



takes 11 minutes to update 2009-today data of all tickers on sec database

#https://pypi.org/project/secfsdstools/
"""
############secfsdstools



from secfsdstools.update import update

from secfsdstools.e_collector.companycollecting import CompanyReportCollector
from secfsdstools.e_filter.rawfiltering import ReportPeriodRawFilter, MainCoregRawFilter, OfficialTagsOnlyRawFilter, USDOnlyRawFilter
from secfsdstools.f_standardize.bs_standardize import BalanceSheetStandardizer
from secfsdstools.f_standardize.cf_standardize import CashFlowStandardizer
from secfsdstools.f_standardize.is_standardize import IncomeStatementStandardizer
import requests
import pandas as pd
from tqdm import tqdm
def secfsdstools():

    ###################
    update()




    ######################################


    headers = {'User-Agent': "juancassinerio@gmail.com"}
    companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json",headers=headers)
    company_df = pd.DataFrame.from_dict(companyTickers.json(),orient='index')

    #tickers amount ordered by marketcap
    n=2
    company_df=company_df.head(n)

    fin=pd.DataFrame()

    for row in tqdm(company_df.itertuples()):
        print(row)


        #standarized df to have common df structure between tickers


        bag = CompanyReportCollector.get_company_collector(ciks=[row.cik_str]).collect() #Microsoft, Alphabet, Amazon
        filtered_bag = bag[ReportPeriodRawFilter()][MainCoregRawFilter()][OfficialTagsOnlyRawFilter()][USDOnlyRawFilter()]
        joined_bag = filtered_bag.join()

        standardizer_bs = BalanceSheetStandardizer()
        standardizer_is = IncomeStatementStandardizer()
        standardizer_cf = CashFlowStandardizer()
        standardized_bs_df = joined_bag.present(standardizer_bs).drop(['adsh','fp','coreg','report', 'cik', 'name', 'fye', 'fy', 'ddate', 'qtrs'], axis=1)


        standardized_is_df = joined_bag.present(standardizer_is).drop(['adsh','fp','coreg','report', 'cik', 'name', 'fye', 'fy', 'ddate', 'qtrs'], axis=1)


        standardized_cf_df = joined_bag.present(standardizer_cf).drop(['adsh','fp','coreg','report', 'cik', 'name', 'fye', 'fy', 'ddate', 'qtrs'], axis=1)

        merged_df = pd.merge(standardized_bs_df, standardized_is_df, on='date', how='outer')
        merged_df = pd.merge(merged_df, standardized_cf_df, on='date', how='outer').drop(['form_x','filed_x','form_y','filed_y'], axis=1)

        merged_df['ticker']=row.ticker
        merged_df['source'] = 'secfsd'

        fin = pd.concat([fin, merged_df])

    fin.to_csv(cwd.parent / 'data' / 'output' / 'df_cme_synthetic.csv')




standardized_bs_df.drop columns adsh cik name qtrs report

import os
target_path = "standardized/BS"
os.makedirs(target_path, exist_ok=True)

standardized_bs_df.save(target_path)




    return

###

for cik in ciklist

merge by date

save into csv

# latest

merge , add dates not present and ticker and source column
df with all tickers

if present secfdf dropdrop yf

else if present yf hold

else add yf

final dataframe stage
end

'''

############SEC API direct##############

#libraries
import requests
import pandas as pd

headers = {'User-Agent': "juancassinerio@gmail.com"}
companyTickers = requests.get("https://www.sec.gov/files/company_tickers.json",headers=headers)
companyData = pd.DataFrame.from_dict(companyTickers.json(),orient='index')
companyData['cik_str'] = companyData['cik_str'].astype(str).str.zfill(10) #tickers
c=0
cik = companyData['cik_str'].iloc[c] #Microsoft # CIK company identificator (ex:APPLE is 320193)
ticker = companyData['ticker'].iloc[c]


#tickers amount ordered by marketcap
n=50
ticker_list = companyData.iloc[:n]['ticker'].tolist()


ticker='AAPL'
cik = companyData[companyData['ticker'] == ticker]['cik_str'].iloc[0]
variables = list(requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',headers=headers).json()['facts']['us-gaap'].keys())
first_variable = variables[84]
first_variable = 'NetCashProvidedByUsedInOperatingActivities'
companyConcept = requests.get((f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}'f'/us-gaap/{first_variable}.json'),headers=headers)


units_dict = companyConcept.json()['units']  # Get the units dictionary

df_units = pd.DataFrame.from_dict(units_dict, orient='index').T
df_units

max_length = max(len(arr) for arr in units_dict.values())
filtered_dict = {key: value for key, value in units_dict.items() if len(value) == max_length}
data = pd.DataFrame.from_dict(filtered_dict)


data10Q = pd.DataFrame.from_dict((companyConcept.json()['units']))
data10Q = pd.json_normalize(data10Q[data10Q.columns[0]])
data10Q = data10Q[data10Q.form == '10-K'] #10-Q / 10-K
data10Q = data10Q.reset_index(drop=True)
data10Q = data10Q[['end', 'val']].drop_duplicates()
data10Q.drop_duplicates(subset=['end'], keep='last', inplace=True)
data10Q = data10Q.rename(columns={'end': 'date', 'val': first_variable})
data10Q.plot(x='date', y=first_variable)


i=0
for ticker in ticker_list:
    cik = companyData[companyData['ticker'] == ticker]['cik_str'].iloc[0]
    #variableslist = ['Assets','AssetsCurrent','PaymentsOfDividendsCommonStock','EffectiveIncomeTaxRateContinuingOperations','CashCashEquivalentsAndShortTermInvestments','CommonStockSharesOutstanding','LiabilitiesCurrent','Liabilities','LongTermDebtNoncurrent','PropertyPlantAndEquipmentNet','NetIncomeLoss']
    #operating cash flow= net income + depr (-def taxes + stopck based compensation +...) - Change WCapital(AC-PC-Cash)
    # 9
    i=i+1
    #variableslist = ['NetCashProvidedByUsedInOperatingActivities','PropertyPlantAndEquipmentNet','CashAndCashEquivalentsAtCarryingValue','CommonStockSharesOutstanding','LiabilitiesCurrent','LiabilitiesNoncurrent','Liabilities','LongTermDebtNoncurrent','InterestExpense','EffectiveIncomeTaxRateContinuingOperations']
    variableslist = ['NetCashProvidedByUsedInOperatingActivities']
    result_df = pd.DataFrame()
    
    for variable in variableslist:
        try:
            companyConcept = requests.get(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{variable}.json', headers=headers)
            companyConcept.raise_for_status()  # Raises exception for bad requests
            units_dict = companyConcept.json()['units']  # Get the units dictionary
            max_length = max(len(arr) for arr in units_dict.values())
            filtered_dict = {key: value for key, value in units_dict.items() if len(value) == max_length}
            data = pd.DataFrame.from_dict(filtered_dict)
            
            #data = pd.DataFrame.from_dict((companyConcept.json()['units']))
            data = pd.json_normalize(data[data.columns[0]])
            if variable == 'EffectiveIncomeTaxRateContinuingOperations':
                data = data[(data.form == '10-K') | (data.form == '20-F')] 
            else:
                data = data[(data.form == '10-K') | (data.form == '20-F')] 
            data = data.reset_index(drop=True)
            data = data[['end', 'val']].drop_duplicates()
            data.drop_duplicates(subset=['end'], keep='last', inplace=True)
            data = data.rename(columns={'end': 'date', 'val': variable})
            if result_df.empty:
                result_df = data.copy()
            else:
                result_df = pd.merge(result_df, data, on='date', how='left')
        except requests.exceptions.RequestException:
            continue
    #if 'LiabilitiesCurrent' in result_df.columns and 'LiabilitiesNoncurrent' in result_df.columns:
     #   result_df['Liabilities'] = result_df['LiabilitiesCurrent'] + result_df['LiabilitiesNoncurrent']
    
    #result_df['date'] = pd.to_datetime(result_df['date'])
    #result_df['Year'] = result_df['date'].dt.year   
          
    #result_df['kd']=result_df['InterestExpense']/result_df['LongTermDebtNoncurrent']
    #result_df['NetDebt']=result_df['Liabilities']-result_df['CashAndCashEquivalentsAtCarryingValue']
    #result_df['DebtCost']=result_df['NetDebt']*(1-result_df['EffectiveIncomeTaxRateContinuingOperations'])
    result_df=result_df.dropna()
    
    #print(i,ticker)
    
    folder_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/financials.xlsx'
    #writer = pd.ExcelWriter(folder_path0, engine='openpyxl', mode='a', if_sheet_exists='replace')
    #if ticker in writer.sheets: # If sheet exist
     #   result_df.to_excel(writer, sheet_name=ticker, startrow=0, startcol=0)  
    #else: # If sheet doesn't exist, add it
     #   result_df.to_excel(writer, sheet_name=ticker)
    #writer.save()
    with pd.ExcelWriter(folder_path0, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        if ticker in writer.sheets:  # If sheet exists
            result_df.to_excel(writer, sheet_name=ticker, startrow=0, startcol=0)
        else:  # If sheet doesn't exist, add it
            result_df.to_excel(writer, sheet_name=ticker)

    print(i, ticker)



ticker='TM'
cik = companyData[companyData['ticker'] == ticker]['cik_str'].iloc[0]
variableslist = ['NetCashProvidedByUsedInOperatingActivities']
result_df = pd.DataFrame()
variables = list(requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',headers=headers).json()['facts']['us-gaap'].keys())
for variable in variableslist:
    try:
        companyConcept = requests.get(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{variable}.json', headers=headers)
        companyConcept.raise_for_status()  # Raises exception for bad requests
        data = pd.DataFrame.from_dict((companyConcept.json()['units']))
        data = pd.json_normalize(data[data.columns[0]])
        if variable == 'EffectiveIncomeTaxRateContinuingOperations':
            data = data[data.form == '10-K'] 
        else:
            data = data[data.form == '10-K'] 
        data = data.reset_index(drop=True)
        data = data[['end', 'val']].drop_duplicates()
        data.drop_duplicates(subset=['end'], keep='last', inplace=True)
        data = data.rename(columns={'end': 'date', 'val': variable})
        if result_df.empty:
            result_df = data.copy()
        else:
            result_df = pd.merge(result_df, data, on='date', how='left')
    except requests.exceptions.RequestException:
        continue
    
    
if 'LiabilitiesCurrent' in result_df.columns and 'LiabilitiesNoncurrent' in result_df.columns:
    result_df['Liabilities'] = result_df['LiabilitiesCurrent'] + result_df['LiabilitiesNoncurrent']

result_df['date'] = pd.to_datetime(result_df['date'])
result_df['Year'] = result_df['date'].dt.year   
      
result_df['kd']=result_df['InterestExpense']/result_df['LongTermDebtNoncurrent']
result_df['NetDebt']=result_df['Liabilities']-result_df['CashAndCashEquivalentsAtCarryingValue']
result_df['DebtCost']=result_df['NetDebt']*(1-result_df['EffectiveIncomeTaxRateContinuingOperations'])

ticker='TM'
cik = companyData[companyData['ticker'] == ticker]['cik_str'].iloc[0]
variableslist = ['NetCashProvidedByUsedInOperatingActivities']
result_df = pd.DataFrame()
variables = list(requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',headers=headers).json()['facts']['us-gaap'].keys())
for variable in variableslist:
    try:
        companyConcept = requests.get(f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{variable}.json', headers=headers)
        companyConcept.raise_for_status()  # Raises exception for bad requests
        data = pd.DataFrame.from_dict((companyConcept.json()['units']))
        data = pd.json_normalize(data[data.columns[0]])
        if variable == 'EffectiveIncomeTaxRateContinuingOperations':
            data = data[data.form == '10-K'] 
        else:
            data = data[data.form == '10-K'] 
        data = data.reset_index(drop=True)
        data = data[['end', 'val']].drop_duplicates()
        data.drop_duplicates(subset=['end'], keep='last', inplace=True)
        data = data.rename(columns={'end': 'date', 'val': variable})




#####################

'''

