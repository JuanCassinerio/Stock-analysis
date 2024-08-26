"""
https://www.youtube.com/watch?v=Wr1NoM3JkTo&t=3s&ab_channel=AdamGetbags
https://www.sec.gov/edgar/sec-api-documentation
https://www.sec.gov/os/webmaster-faq#developers
"""

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


ticker='TM'
cik = companyData[companyData['ticker'] == ticker]['cik_str'].iloc[0]
variables = list(requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',headers=headers).json()['facts']['us-gaap'].keys())
first_variable = variables[84]
first_variable = 'NetCashProvidedByUsedInOperatingActivities'
companyConcept = requests.get((f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}'f'/us-gaap/{first_variable}.json'),headers=headers)


units_dict = companyConcept.json()['units']  # Get the units dictionary
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



