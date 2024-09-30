


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














