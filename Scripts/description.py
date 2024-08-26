import pandas as pd
import yfinance as yf

file_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/fin.csv'
file_path1 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/description.csv'


tickers = pd.read_csv(file_path0)['ticker'].unique().tolist()

df = pd.DataFrame(columns=["ticker","currency","Company","industry","sector","country","Summary","link_yf"])


for ticker in tickers:
        company_info = yf.Ticker(ticker).info
        currency = company_info.get("currency", None)
        shortName = company_info.get("shortName", None)
        industry = company_info.get("industry", None)
        sector = company_info.get("sector", None)
        country = company_info.get("country", None)
        longBusinessSummary = company_info.get("longBusinessSummary", None)
        link_yf = f"https://finance.yahoo.com/quote/{ticker}?.tsrc=fin-srch"

        # Append information to DataFrame (handle potential missing values)
        df = df.append({
            "ticker": ticker,
            "currency": currency,
            "Company": shortName,
            "industry": industry,
            "sector": sector,
            "country": country,
            "Summary": longBusinessSummary,
            "link_yf": link_yf,
        }, ignore_index=True)
 
    
df.to_csv(file_path1, index=False)

'''
data.news
data.actions #dividens and splits  data.dividends

previousClose': 415.81,
'open': 417.9,
'dayLow': 417.27,
'dayHigh': 422.085,
'currentPrice': 421.46,

# 'volume': 8472120,
'''