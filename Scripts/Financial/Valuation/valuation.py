import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import requests
import re
from bs4 import BeautifulSoup

################## VALUATION

def salesprojection_logex(t, g, rev_0, a, b, c):
    return g + a / (1 + np.exp(-b * (t - c)))*(1-a / (1 + np.exp(-b * (t - c))))


def salesprojection_logex_fast_short(t, g, rev_0, a, b, c):
    return g + a / (1 + np.exp(-b * (t - c)))*(1-a / (1 + np.exp(-b * (t - c))))

def salesprojection_logex_fast_long(t, g, rev_0, a, b, c):
    return g + a / (1 + np.exp(-b * (t - c)))*(1-a / (1 + np.exp(-b * (t - c))))

def salesprojection_exfall_rise_other(t,g, a, b):
    return g + np.exp(-a * (t - b))

def salesprojection_exfall_rise(t, g, b, c):     #scales strats with ti=0
    return g + b * np.exp(-c * (t))

def aprox(value, target, tolerance):
    return (target - tolerance) <= value <= (target + tolerance)


def cost_of_equity(stock,macros,rf,ERP): #5 years monthly
    # https://www.alphaspread.com/security/nasdaq/aapl/discount-rate
    # https://site.financialmodelingprep.com/weighted-average-cost-of-capital/AAPL
    # https://valueinvesting.io/AAPL/valuation/wacc
    #stock=ticker_data['price']
    spy=macros['SPY']
    stock['Month'] = stock['Date'].dt.month
    stock['Year'] = stock['Date'].dt.year
    df = pd.merge(spy, stock, on='Date')
    df = df.rename(columns={'Adj Close_x': 'SPY', 'Adj Close_y': 'price'})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.resample('ME').ffill()
    df['SPY'] = df['SPY'].pct_change()
    df['price_change'] = df['price'].pct_change()
    df = df.dropna()
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    window = 5 * 12
    df['cov'] = df['SPY'].rolling(window=window).cov(df['price_change'])
    df['var'] = df['SPY'].rolling(window=window).var()
    df['beta'] = df['cov'] / df['var']
    df = df[['Date','price','beta']]
    beta = df['beta'].iloc[-1]
    ke = rf+beta*ERP
    return ke

def damodaran_1(ticker_data):  # yahooinput
    ticker = 'APPL'
    data = financial_statements

    for col in data.columns:
        if col != 'Date':
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to float, coerce invalid to NaN

    data['Year'] = data['Date'].dt.year
    data = data.dropna(subset=['Revenue'])

    # project revenues (scaled data)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    years_scaled = scaler_x.fit_transform(data['Year'].values.reshape(-1, 1)).flatten()
    revenue_scaled = scaler_y.fit_transform(data['Revenue'].values.reshape(-1, 1)).flatten()
    popt, _ = curve_fit(salesprojection, years_scaled, revenue_scaled, maxfev=100)

    data = data.sort_values(by='Date', ascending=True)
    data['generated'] = 0

    data.drop('Year', axis=1)

    # Vertical Analysis to Revenue
    variables = ['Net Income', 'Reconciled Depreciation', 'Net PPE', 'Current Assets', 'Total Non Current Assets',
                 'Current Liabilities', 'Total Non Current Liabilities Net Minority Interest',
                 'Cash And Cash Equivalents']

    verticalratio = {
        variable: pd.DataFrame({variable: data[variable] / data['Total Revenue'] for variable in variables})[
            variable].mean() for variable in variables}

    # wacc scraping
    url = f"https://valueinvesting.io/{ticker}/valuation/wacc"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}  # Mimic a real browser to avoid potential blocking
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")  # this contains the info
    wacc_element = soup.find("meta", {"name": "description"})
    wacc_content = wacc_element["content"]
    wacc_value_start = wacc_content.find("The WACC of") + len("The WACC of")
    wacc_value_end = wacc_content.find("%", wacc_value_start)
    wacc = float(re.search(r"([\d.]+)", wacc_content[wacc_value_start:wacc_value_end].strip()).group(1))

    # other variables
    Years_Depreciation = (data['Net PPE'] / data['Reconciled Depreciation']).mean()
    Net_Debt = (data['Current Liabilities'] + data['Total Non Current Liabilities Net Minority Interest'] - data[
        'Cash And Cash Equivalents']).iloc[-1]
    shares = data['Ordinary Shares Number'].iloc[-1]

    # fcf Projection



    Datelast_date = data['Date'].iloc[-1]

    future_years = pd.date_range(start=Datelast_date + pd.DateOffset(years=1), periods=5, freq='YE')
    future_years_scaled = scaler_x.transform(future_years.year.values.reshape(-1, 1)).flatten()

    revenues = scaler_y.inverse_transform(salesprojection(future_years_scaled, *popt).reshape(-1, 1))

    net_incomes = revenues * verticalratio['Net Income']
    current_assets = revenues * verticalratio['Current Assets']
    current_liabilities = revenues * verticalratio['Current Liabilities']
    cash = revenues * verticalratio['Cash And Cash Equivalents']
    total_non_current_assets = revenues * verticalratio['Total Non Current Assets']
    total_non_current_liabilities = revenues * verticalratio['Total Non Current Liabilities Net Minority Interest']
    net_pp = revenues * verticalratio['Net PPE']
    depreciation = net_pp / Years_Depreciation

    revenues = revenues.flatten()
    net_incomes = net_incomes.flatten()
    current_assets = current_assets.flatten()
    current_liabilities = current_liabilities.flatten()
    cash = cash.flatten()
    total_non_current_assets = total_non_current_assets.flatten()
    total_non_current_liabilities = total_non_current_liabilities.flatten()
    net_pp = net_pp.flatten()
    depreciation = depreciation.flatten()

    # Create a DataFrame for new data
    new_year_data = {
        'Date': future_years,
        'Total Revenue': revenues,
        'Net Income': net_incomes,
        'Current Assets': current_assets,
        'Current Liabilities': current_liabilities,
        'Cash And Cash Equivalents': cash,
        'Total Non Current Assets': total_non_current_assets,
        'Total Non Current Liabilities Net Minority Interest': total_non_current_liabilities,
        'Net PPE': net_pp,
        'Reconciled Depreciation': depreciation,
        'generated': 1
    }

    new_df = pd.DataFrame(new_year_data)
    data = pd.concat([data, new_df], ignore_index=True)

    # FCFF
    Operatingcashflow = data['Net Income'] + data['Reconciled Depreciation']
    Capex = data['Net PPE'] - data['Net PPE'].shift(1) + data['Reconciled Depreciation']
    NWCCh = (data['Current Assets'] - data['Current Liabilities'] - data['Cash And Cash Equivalents']) - (
            data['Current Assets'] - data['Current Liabilities'] - data['Cash And Cash Equivalents']).shift(1)
    data['Free Cash Flow'] = Operatingcashflow - Capex - NWCCh

    g = 3

    fcfnext = data['Free Cash Flow'].iloc[-1] * (1 + g / 100)
    terminalvalue = fcfnext / ((wacc / 100) - (g / 100))
    Subtotal = data['Free Cash Flow'].tolist()
    Subtotal[-1] += terminalvalue

    npv=npv_function(cash_flows, wacc, g)

    VA_Asset = npv(Subtotal[-5:], wacc, g)
    VA_Equity = VA_Asset - Net_Debt
    TarjetPrice_mean = VA_Equity / shares

    return TarjetPrice_mean, wacc


def damodaran_2(ticker_data,macros):
    ############################## 1 - DATA PREPARATION ##############################
    #data =result_df
    ticker = ticker_data['description']['ticker']
    data = ticker_data['financial_statements']

    data = data.drop_duplicates(subset='Date', keep='first')
    data=data.head(16) #last 4 years of Q data
    data['Date Unix'] = data['Date'].astype(np.int64) // 10**9
    data['Year'] = data['Date'].dt.year
    data = data.sort_values(by='Date Unix', ascending=True)
    data = data.dropna(subset=['Revenue'])
    data = data[data['Revenue'] >= 0] #Deal with negative revenues that crash revenue change
    data['Revenue Change'] = data['Revenue'].pct_change(periods=4)
    data = data.dropna(subset=['Revenue Change'])

    ############################## 2 - Vertical Analysis to Revenue ##############################
    variables = ['Net Income', 'Depreciation', 'Capex', 'PPE', 'Assets Current', 'Liabilities Current',
                 'Long Term Debt', 'Cash', 'Operating Cash Flow']
    verticalratio = {
        variable: pd.DataFrame({variable: data[variable] / data['Revenue'] for variable in variables})[variable].mean()
        for variable in variables}

    ############################## 3 - REVENUE FITTING ##############################
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    date_scaled = scaler_x.fit_transform(data['Date Unix'].values.reshape(-1, 1)).flatten()

    '''functions = [salesprojection_logex,salesprojection_exfall_rise]
    max_r2 = -np.inf
    best_fit_params = None
    best_function = None

    for func in functions:
        popt, _ = curve_fit(func, years_scaled, revenue_scaled, p0=[g, rev_0, t_0, a, b, c], maxfev=100)
        y_pred = func(years_scaled, *popt)
        r2 = r2_score(revenue_scaled, y_pred)

        if r2 > max_r2:
            max_r2 = r2
            best_fit_params = popt
            best_function = func
            
            def salesprojection_exfall_rise(t,a,b,c):
    return a + np.exp(-b * (t - c))
            
    '''

    data['Smoothed Revenue Change'] = data['Revenue Change'].ewm(span=8, adjust=False).mean()

    g = (1 + macros['g'] / 100) ** (1 / 4) - 1  # quaterly
    g_d = (1 + macros['g_std'] / 100) ** (1 / 4) - 1

    b = data['Smoothed Revenue Change'].iloc[0]-g

    def best_function_fixed(t, c): #long term g fixing
        return salesprojection_exfall_rise(t, g, b, c)


    best_fit_params, covariance = curve_fit(best_function_fixed, date_scaled, data['Smoothed Revenue Change'],maxfev=100000)
    c_d=np.sqrt(np.diag(covariance)) #slope margin
    y_pred = best_function_fixed(date_scaled, *best_fit_params)
    max_r2 = r2_score(data['Smoothed Revenue Change'], y_pred) * 100
    print(max_r2,"%")

    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_scaled, y=data['Smoothed Revenue Change'], name='Smoothed Revenue Change' ))
    fig.add_trace(go.Scatter(x=date_scaled, y=data['Revenue Change'], name='Revenue Change' ))
    fig.add_trace(go.Scatter(x=date_scaled, y=y_pred,line=dict(dash='dash'), name='Revenue Trend' ))  # Set line dash style to 'dash'
    fig.show()
    '''


    ############################## 4 - VARIABLES Calculation ##############################¿

    data = data.sort_values(by='Date', ascending=True)
    shares = data['Shares'].iloc[-1]
    Marketcap = ticker_data['price']['Adj Close'].iloc[0] * shares
    EA_ratio = Marketcap / (data['Long Term Debt'].iloc[-1] + Marketcap)

    Years_Depreciation = (data['PPE'] / data['Depreciation']).mean()
    Net_Debt = (data['Liabilities Non Current'] - data['Cash and ST Investments']).iloc[-1]


    ############################## 5 - CF AND WACC Projection_Scenarios ##############################

    data['generated'] = 0

    rf,rf_std=macros['rf'],macros['rf_std']
    ERP,ERP_std=macros['ERP'],macros['ERP_std']

    scenarios = {
    'optimistic': (best_fit_params - c_d, g, rf - rf_std,ERP-ERP_std),
    'normal': (best_fit_params, g, rf,ERP),
    'pessimistic': (best_fit_params + c_d, g , rf + rf_std,ERP+ERP_std)
    }

    scenarios_df = pd.DataFrame()
    
    for scenario_name, (c_value, growth_rate, risk_free_rate,ERP) in scenarios.items():
        scenario_data = pd.DataFrame({'Scenario': [scenario_name],'c': [c_value],'g': [growth_rate],'rf': [risk_free_rate],'ERP': [ERP]})

        Date_last = data['Date'].iloc[-1]
        future_date = Date_last
        revenue_last = data['Revenue'].iloc[-1]
        revenue = revenue_last

        scenario_name=scenario_data['Scenario'][0]
        g=scenario_data['g'][0]
        b=data['Smoothed Revenue Change'].iloc[0]-g
        c=scenario_data['c'][0][0]
        rf=scenario_data['rf']
        ERP=scenario_data['ERP']

        def best_function_fixed(t, c): #long term g fixing
            return salesprojection_exfall_rise(t, g, b, c)

        data_copy = data.copy()

        ke=cost_of_equity(ticker_data['price'],macros,rf,ERP)
        wacc=ke*EA_ratio+data_copy['kd'].iloc[-1]*100*(1-data_copy['tax'].iloc[-1])*(1-EA_ratio)

        revenues=[]
        future_dates=[]
        future_dates_Unix=[]

        for i in range(0, 20):
            if i == 20:
                error = 'max iterations reached'
                break  # Replace `exit()` with `break` to avoid terminating the whole program

            # Calculate the next future date
            future_date += pd.DateOffset(months=4)  # Increment date by one year 1 Q!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            future_date_Unix = int(future_date.timestamp())
            future_date_Unix_scaled = scaler_x.transform(np.array([[future_date_Unix]]))[0][0]

            # Predict the revenue change and unscale the predicted value
            revenue_change = best_function_fixed(future_date_Unix_scaled, *best_fit_params)

            # Update the revenue with the predicted change
            revenue *= 1 + revenue_change

            # Append the values to the corresponding lists
            future_dates.append(future_date)
            future_dates_Unix.append(future_date_Unix)
            revenues.append(revenue)


            # Check if the revenue change approximation is close enough to stop

            if aprox(revenue_change, g, g_d):
                break

        revenues = np.array(revenues)

        net_incomes = (revenues * verticalratio['Net Income']).flatten()
        current_assets = (revenues * verticalratio['Assets Current']).flatten()
        current_liabilities = (revenues * verticalratio['Liabilities Current']).flatten()
        cash = (revenues * verticalratio['Cash']).flatten()
        net_pp = (revenues * verticalratio['PPE']).flatten()
        Capex = (revenues * verticalratio['Capex']).flatten()
        OperatingCashFlow = (revenues * verticalratio['Operating Cash Flow']).flatten()

        depreciation = (net_pp / Years_Depreciation).flatten()

        projected_data = {'Date': future_dates, 'Date Unix': future_dates_Unix, 'Revenue': revenues,
                          'Net Income': net_incomes, 'Assets Current': current_assets,
                          'Liabilities Current': current_liabilities, 'Cash': cash, 'PPE': net_pp, 'Capex': Capex,
                          'Operating Cash Flow':OperatingCashFlow,
                          'Depreciation': depreciation, 'generated': 1}
        new_df = pd.DataFrame(projected_data)
        data_copy = pd.concat([data_copy, new_df], ignore_index=True)

        # FCFF
        #Operatingcashflow = data['Net Income'] + data['Depreciation']
        Operatingcashflow = data_copy['Operating Cash Flow']
        Capex = data_copy['Capex'] #+ data['Depreciation']
        #NWCCh = (data['Assets Current'] - data['Liabilities Current'] - data['Cash']) - (data['Assets Current'] - data['Liabilities Current'] - data['Cash']).shift(1)
        data_copy['Free Cash Flow'] = Operatingcashflow - Capex #- NWCCh

        fcfnext = data_copy['Free Cash Flow'].iloc[-1] * (1 + g)
        terminalvalue = fcfnext *(1+(((1 + wacc / 100) ** (1 / 4) - 1)))/ ((((1 + wacc / 100) ** (1 / 4) - 1)) - g)
        Subtotal = data_copy['Free Cash Flow'].tolist()
        Subtotal.append(terminalvalue)

        def npv(cash_flows, wacc):
            cash_flows=Subtotal[-(len(new_df) + 1):]
            npv = 0
            for t, cash_flow in enumerate(cash_flows):
                t += 1
                npv += cash_flow / (1 + (((1 + wacc / 100) ** (1 / 4) - 1))) ** t # Q 1/4 quaterly correction
            return npv

        VA_Asset = npv(Subtotal[-(len(new_df) + 1):], wacc)
        VA_Equity = VA_Asset[0] - Net_Debt
        TarjetPrice_mean = VA_Equity / shares

        results={'scenario':scenario_name,'Tarjet Price':TarjetPrice_mean}
        scenarios_df = pd.concat([scenarios_df, pd.DataFrame([results])], ignore_index=True)

    print(scenarios_df)

    results = {'Date_t0': Date_last, 'TarjetPrice_scenarios': scenarios_df, 'R2': max_r2,
               'Projected Financial Statements': data}
    '''
    
    #https://valueinvesting.io/AAPL/valuation/wacc
    salesprojection_logex(t, g, rev_0, a, b, c)
    future_years
    best_fit_params, _ = curve_fit(best_function, years_scaled, revenue_scaled, p0=[g, rev_0, t_0, a, b, c], maxfev=100)
    y_pred = best_function(years_scaled, *best_fit_params)
    
    
        logex_variables={equity_ratio,debt_ratio,kd,t,beta}
        for variable in logex_variables:
            #fit
            best_fit_params, _ = curve_fit(salesprojection_logex, years_scaled, revenue_scaled, p0=[g, rev_0, t_0, a, b, c], maxfev=100)
            #predict
            y_pred = salesprojection_logex(years_scaled, *best_fit_params)
    
            append
    
    
        wacc = ke * (Marketcap / data['Assets']) + data['Assets'] * (1 - data['tax']) * (data['Debt'] / data['Assets'])
    '''


    # margin errors


    '''
    # Today valuation
    fcfnext0 = data['Free Cash Flow'].iloc[-2] * (1 + g / 100)
    terminalvalue0 = fcfnext0 / ((wacc / 100) - (g / 100))
    Subtotal0 = data['Free Cash Flow'].tolist()
    del Subtotal0[:4]
    del Subtotal0[-1]
    Subtotal0.append(terminalvalue0)
    VA_Asset = npv(Subtotal0, wacc)
    VA_Equity = VA_Asset - Net_Debt
    TarjetPrice_0today = VA_Equity / shares

    # +1 year valuation
    fcfnext1 = data['Free Cash Flow'].iloc[-1] * (1 + g / 100)
    terminalvalue1 = fcfnext1 / ((wacc / 100) - (g / 100))
    Subtotal1 = data['Free Cash Flow'].tolist()
    del Subtotal1[:5]
    Subtotal1.append(terminalvalue1)
    VA_Asset = npv(Subtotal1, wacc)
    VA_Equity = VA_Asset - Net_Debt
    TarjetPrice_1yplus = VA_Equity / shares'''

    #results= {'Date_t0':Date_last,'TarjetPrice_t0':TarjetPrice_0today, 'TarjetPrice_t1':TarjetPrice_1yplus,'R2':max_r2,'Fitting function':best_function,'Fitting params':best_fit_params,'Projected Financial Statements':data}

    return results



