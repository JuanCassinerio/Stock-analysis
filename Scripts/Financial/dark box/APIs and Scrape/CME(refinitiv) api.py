# This script downloads historical data using the eikon package for the
# main futures contracts traded on CME.


#save data to csv on my pc, upload data to github or gdrive
#streamlit webpage with commoditie selction grpah and data downloader botton .csv format + description


import pandas as pd
import eikon as ek
from pathlib import Path


def date_tag(year: int, month: int):
    """
    Returns the time part and expiry condition of a RIC for year-month pairs.
    :param year: full year of expiry (e.g. 2021)
    :param month: number of month of expiry (1 to 12)
    :return: a tuple of the date part of the RIC, plus a boolean indicating if
    it is expired or not. If the expiry condition is unknown, it returns
    both possible cases.
    """
    today = pd.to_datetime('today')
    if year < today.year or (year == today.year and month < today.month):
        ric = [f'{month_symbols[month]}{str(year)[3]}^{str(year)[2]}']
    elif year > today.year:
        ric = [f'{month_symbols[month]}{str(year)[-2:]}']
    elif (year == today.year) and (month > today.month):
        ric = [f'{month_symbols[month]}{str(year)[-1]}',
               f'{month_symbols[month]}{str(year)[-2:]}']
    elif (year == today.year) and (month == today.month):
        ric = [f'{month_symbols[month]}{str(year)[-1]}',
               f'{month_symbols[month]}{str(year)[3]}^{str(year)[2]}']
    else:
        raise ValueError('Could not parse')
    return [ric[i] for i in range(len(ric))]


# Create folder to store downloaded data
cwd = Path.cwd()
parent = cwd.parent
tmp_path = parent / 'data' / 'tmp'
tmp_path.mkdir(parents=True, exist_ok=True)

product_symbols = {'SOJA': 'S', 'MAIZ': 'C'}

month_symbols = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

months = {'SOJA': [1, 3, 5, 7, 9, 11], 'MAIZ': [3, 5, 7, 9, 12]}

# Build list of all relevant RIC (by combining products with relevant months)
all_ric_list = []
for year in range(2024, pd.to_datetime('today').year + 4):
    for symb in product_symbols.keys():
        for month in months[symb]:
            tag = date_tag(year, month)
            for i in range(len(tag)):
                all_ric_list.append(product_symbols[symb] + tag[i])

fields = ['TR.SETTLEMENTPRICE.Date', 'TR.SETTLEMENTPRICE']

ek.set_app_key('6afaf18fbb04470a9451679d17ee4ffb69899092')

df, e = ek.get_data(instruments=all_ric_list,
                    fields=fields,
                    parameters={'Sdate': '2018-01-01',
                                'Edate': pd.to_datetime(
                                    'today').strftime('%Y-%m-%d')})

df.to_csv(tmp_path / 'downloaded_cme_series_eikon_raw.csv', sep=';',
          index=False)

import pandas as pd
from pathlib import Path

cwd = Path.cwd()
parent = cwd.parent.absolute()
tmp_path = parent.joinpath('data', 'tmp')
rdp_data = tmp_path.joinpath('curated_cme_series_rdp_raw.csv')
ek_data = tmp_path.joinpath('downloaded_cme_series_eikon_raw.csv')
output_path = parent.joinpath('data', 'output')
shared_path = output_path.joinpath('shared')
shared_path.mkdir(parents=True, exist_ok=True)


def expiry_from_symbol(row):
    symbol = row['UnderlyingSymbol']
    if '^' in symbol[-3:]:
        year = 2000 + int(symbol[-1]) + 10 * int(symbol[-3])
        month = symbol_months[symbol[-4]]
    else:
        year = 2000 + int(symbol[-2:])
        month = symbol_months[symbol[-3]]
    return year, month


df_ek = pd.read_csv(ek_data, parse_dates=['Date'], sep=';')

product_id = {'SOJA': 5, 'TRIGO': 9, 'MAIZ': 1}

months = {'SOJA': [1, 3, 5, 7, 9, 11], 'MAIZ': [3, 5, 7, 9, 12]}

month_symbols = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
symbol_months = {v: k for k, v in month_symbols.items()}

df_ek = df_ek.rename(columns={'Instrument': 'UnderlyingSymbol',
                              'Settlement Price': 'SettlePrice'})

df_ek = df_ek.dropna(subset=['SettlePrice'])

df_ek['ProductId'] = df_ek['UnderlyingSymbol'].str[0].map(
    {'S': product_id['SOJA'], 'C': product_id['MAIZ']})

df_ek[['ExpiryYear', 'ExpiryMonth']] = df_ek.apply(
    lambda x: expiry_from_symbol(x), axis=1, result_type='expand')

df_ek['MktId'] = 2
df_ek['CurrencyId'] = 2
df_ek['PortId'] = 2
df_ek['SettlePrice'] = df_ek['SettlePrice'] / 100  # Convert cents to USD
df_ek['QuantityUnitId'] = 4  # Proposed t_mae_UnidadMedida ID value for bushel

df_ek['Date'] = df_ek['Date'].dt.tz_localize(None)

# Curation process

# Navigate to parent directory containing Fuentes folder
while not parent.joinpath('Fuentes').exists():
    parent = parent.parent.absolute()

holidays_path = parent.joinpath('Fuentes', 'feriados', 'data', 'output',
                                'shared', 'consolidated_holidays.csv')

holidays = pd.read_csv(holidays_path, sep=';', parse_dates=['Date'])
holidays = holidays.loc[holidays['MktCbot'], 'Date']

start_date = '1990-01-01'
end_date = pd.to_datetime('today').strftime('%Y-%m-%d')

full_date_range = pd.date_range(start=start_date, end=end_date)
business_days = pd.bdate_range(start=start_date, end=end_date, freq='C',
                               holidays=list(holidays))

non_business_days = full_date_range.difference(business_days)

ok_list = []
non_business_list = []
missing_price_list = []
missing_dates_list = []

years = df_ek['ExpiryYear'].unique()

# Hay fechas duplicadas, por lo que pude ver mayormente visperas de feriados
df_ek_dup = df_ek[df_ek.duplicated()]
df_ek.drop_duplicates(inplace=True, keep='first')

for prod_id in product_id.values():
    for year in years:
        for month in range(1, 13):
            df = df_ek.loc[(df_ek['ProductId'] == prod_id) &
                           (df_ek['ExpiryYear'] == year) &
                           (df_ek['ExpiryMonth'] == month)]
            if len(df) > 0:
                # Check if any register correspond to a non-business day
                cond_non_business = ~df['Date'].isin(non_business_days)
                df = df.loc[cond_non_business]
                df_non_business = df.loc[~cond_non_business]
                if len(df_non_business) > 0:
                    non_business_list.append(df_non_business)

                # Remove any registers without price
                cond_price = (~(df['SettlePrice'].isnull()) &
                              ~(df['SettlePrice'] == 0))
                df = df.loc[cond_price]
                df_missing_price = df.loc[~cond_price]
                if len(df_missing_price) > 0:
                    missing_price_list.append(df_missing_price)

                # Check if all business days are present in the series

                # Transform the index to consider all business days in the
                # range of the original data
                df = df.set_index(df['Date'])
                df = df.sort_index()
                df_dates = df['Date']  # These are the dates that have data
                df.drop(columns=['Date'], inplace=True)
                df_reidx = df.reindex(business_days)
                df_reidx = df_reidx.loc[df.index[0]:df.index[-1]]
                # The index of df_reidx has all business days in the range

                # We check for differences and save any business days that
                # are missing on the webscraped data
                if df_reidx.index.difference(df_dates).size > 0:
                    df_missing_dates = pd.DataFrame(
                        df_reidx.index.difference(df_dates))
                    df_missing_dates.columns = ['Date']
                    df_missing_dates['UnderlyingSymbol'] = df.iloc[0][
                        'UnderlyingSymbol']
                    missing_dates_list.append(df_missing_dates)

                df.reset_index(inplace=True)
                df = df.sort_values(by='Date')
                ok_list.append(df)

# Diagnostic process

if len(ok_list) > 0:
    output_ok = pd.concat(ok_list)
    output_ok.to_csv(tmp_path.joinpath('curated_cme_historical_series_eikon.csv'),
                     index=False, sep=';', encoding='utf-8-sig')

import pandas as pd
from pathlib import Path

cwd = Path.cwd()
parent = cwd.parent.absolute()
tmp_path = parent.joinpath('data', 'tmp')
output_path = parent.joinpath('data', 'output')
shared_path = output_path.joinpath('shared')
shared_path.mkdir(parents=True, exist_ok=True)
rdp_data = tmp_path.joinpath('curated_cme_historical_series_rdp.csv')
ek_data = tmp_path.joinpath('curated_cme_historical_series_eikon.csv')

df_rdp = pd.read_csv(rdp_data, parse_dates=['Date'], sep=';')

# Drop unused columns
df_rdp.drop(columns=['OpenInterest', 'Volume'], inplace=True)
# Keep reduced number of months
df_rdp = df_rdp.loc[
    ((df_rdp['ProductId'] == 5) & df_rdp['ExpiryMonth'].isin([1, 3, 5, 7, 9, 11]))
    |
    ((df_rdp['ProductId'] == 1) & df_rdp['ExpiryMonth'].isin([3, 5, 7, 9, 12]))].copy()

df_ek = pd.read_csv(ek_data, parse_dates=['Date'], sep=';')

df = pd.concat([df_rdp, df_ek], ignore_index=True)
df = df.sort_values(by=['UnderlyingSymbol', 'Date'])

# Keep reduced number of months
df_rdp = df_rdp.loc[
    ((df_rdp['ProductId'] == 5) & df_rdp['ExpiryMonth'].isin([1, 3, 5, 7, 9, 11]))
    |
    ((df_rdp['ProductId'] == 1) & df_rdp['ExpiryMonth'].isin([3, 5, 7, 9, 12]))].copy()

# Tiro los registros duplicados sin conflicto (mismo simbolo, fecha y precio)
df.drop_duplicates(subset=['UnderlyingSymbol', 'Date', 'SettlePrice'],
                   keep='first', inplace=True)

# Buscamos algun registro duplicado inconsistente, que tengan el mismo simbolo y fecha).
# Por lo que acabamos de hacer, estos solo pueden tener precios distintos y habría que decidir.
dup = df.duplicated(subset=['UnderlyingSymbol', 'Date'], keep=False)
if len(df[dup]) > 0:
    print('Conflicting duplicates found')
    df[dup].to_excel('duplicated_dates.xlsx', index=False)
else:
    print('No conflicting duplicates found')

df.to_csv(shared_path.joinpath('curated_cme_historical_series.csv'), sep=';', index=False)