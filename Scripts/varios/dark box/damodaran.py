
def damodaranfcf(data,ticker,g): 
    # FCF Projection
    data['Year'] = data['Date'].dt.year
    popt, _ = curve_fit(salesprojection, data['Year'], data['fcf'], maxfev=100000000) #p0=[5E+09, - 1E+13]
    
    Datelast_num = data['Date'].dt.year.iloc[-1]
    Datelast_date = data['Date'].iloc[-1]
    for i in range(1, 7):
        Date = Datelast_date + pd.DateOffset(years=1)*i
        FreeCashFlow = salesprojection(Datelast_num + i, *popt)
        new_year_data = {'Date': Date,'fcf': FreeCashFlow}
        data = data.append(new_year_data, ignore_index=True)

    
    # Today valuation
    fcfnext0 = data['fcf'].iloc[-2] * (1+g/100)
    terminalvalue0 = fcfnext0 / ((wacc/100)-(g/100))
    Subtotal0 = data['fcf'].tolist()
    del Subtotal0[:4]
    del Subtotal0[-1]
    Subtotal0.append(terminalvalue0)
    VA_Asset = npv(Subtotal0, wacc,g)
    VA_Equity=VA_Asset-Net_Debt
    TarjetPrice_0today = VA_Equity/shares
    
    # +1 year valuation
    fcfnext1 = data['fcf'].iloc[-1] * (1+g/100)
    terminalvalue1 = fcfnext1 / ((wacc/100)-(g/100))
    Subtotal1 = data['fcf'].tolist()
    del Subtotal1[:5]
    Subtotal1.append(terminalvalue1)
    VA_Asset = npv(Subtotal1, wacc,g)
    VA_Equity=VA_Asset-Net_Debt
    TarjetPrice_1yplus = VA_Equity/shares
    
    return TarjetPrice_0today,TarjetPrice_1yplus,plt



def damodaran(data):    
    data['Year'] = data['Date'].dt.year
    popt, _ = curve_fit(salesprojection, data1['Year'], data1['Total Revenue'], maxfev=1000)
    slope = popt[0]
    intercept = popt[1]
    data1.drop('Year', axis=1, inplace=True)
    
    # Vertical Analysis to Revenue
    variables = ['Net Income','Reconciled Depreciation','Net PPE','Current Assets','Total Non Current Assets', 'Current Liabilities','Total Non Current Liabilities Net Minority Interest', 'Cash And Cash Equivalents']
    verticalratio = {}
    for variable in variables:
        ratio_name = f'{variable} Vertical Ratio'
        data1[ratio_name] = data1[variable] / data1['Total Revenue']
        verticalratio[variable] = data1[ratio_name].mean()
    
    # Other Variables
    url = f"https://valueinvesting.io/{ticker}/valuation/wacc"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}  # Mimic a real browser to avoid potential blocking
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser") #this contains the info
    wacc_element = soup.find("meta", {"name": "description"})
    wacc_content = wacc_element["content"]
    wacc_value_start = wacc_content.find("The WACC of") + len("The WACC of")
    wacc_value_end = wacc_content.find("%", wacc_value_start)
    wacc= float(re.search(r"([\d.]+)", wacc_content[wacc_value_start:wacc_value_end].strip()).group(1))
    
    Years_Depreciation=(data1['Net PPE']/data1['Reconciled Depreciation']).mean()
    Net_Debt=(data1['Current Liabilities']+data1['Total Non Current Liabilities Net Minority Interest']-data1['Cash And Cash Equivalents']).iloc[-1]
    shares=data1['Ordinary Shares Number'].iloc[-1]
    
    # Sales Projection
    Datelast_num = data1['Date'].dt.year.iloc[-1]
    Datelast_date = data1['Date'].iloc[-1]
    for i in range(1, 6):
        Date = Datelast_date + pd.DateOffset(years=1)*i
        Revenue = salesprojection(Datelast_num + i, slope, intercept)
        NetIncome = Revenue * verticalratio['Net Income']
        CurrentAssets = Revenue * verticalratio['Current Assets']
        CurrentLiabilities = Revenue * verticalratio['Current Liabilities']
        Cash = Revenue * verticalratio['Cash And Cash Equivalents']
        TotalNonCurrentAssets = Revenue * verticalratio['Total Non Current Assets']
        TotalNonCurrentLiabilities = Revenue * verticalratio['Total Non Current Liabilities Net Minority Interest']
        NetPP = Revenue * verticalratio['Net PPE']
        Depreciation = NetPP/Years_Depreciation
    
        new_year_data = {'Date': Date,'Net Income': NetIncome, 'Current Assets': CurrentAssets,'Current Liabilities': CurrentLiabilities,'Cash And Cash Equivalents': Cash,  'Total Non Current Assets':
                        TotalNonCurrentAssets, 'Total Non Current Liabilities Net Minority Interest': TotalNonCurrentLiabilities, 'Net PPE': NetPP,'Reconciled Depreciation': Depreciation}
    
        data1 = data1.append(new_year_data, ignore_index=True)
    
    # FCFF
    Operatingcashflow = data1['Net Income'] + data1['Reconciled Depreciation']
    Capex = data1['Net PPE'] - data1['Net PPE'].shift(1) + data1['Reconciled Depreciation']
    NWCCh = (data1['Current Assets']-data1['Current Liabilities']-data1['Cash And Cash Equivalents']) - (data1['Current Assets']-data1['Current Liabilities']-data1['Cash And Cash Equivalents']).shift(1)
    data1['Free Cash Flow'] = Operatingcashflow - Capex - NWCCh
    
    g=3
    
    fcfnext = data1['Free Cash Flow'].iloc[-1] * (1+g/100)
    terminalvalue = fcfnext / ((wacc/100)-(g/100))
    Subtotal = data1['Free Cash Flow'].tolist()
    Subtotal[-1] += terminalvalue
    
    
    def npv(cash_flows, wacc, g):
        npv = 0
        for t, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + (wacc/100)) ** t
        return npv
    
    VA_Asset = npv(Subtotal[-5:], wacc,g)
    VA_Equity=VA_Asset-Net_Debt
    TarjetPrice_mean = VA_Equity/shares
    return TarjetPrice_mean,wacc














