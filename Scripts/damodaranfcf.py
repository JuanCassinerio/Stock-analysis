#libraries
from scipy.optimize import curve_fit
import pandas as pd



#def salesprojection(x,a1,b1,c1,g,b2,c2):
#    return a1*(1+b1/100)**(x-c1)+g/(1+np.exp(b2*(x-c2)))

#def salesprojection(x,a1,b1,c1):
#    return a1*(1+b1/100)**(x-c1)

#1 fcf projection
def fcfprojection(x,a,b):
    return a*x+b

#2 macro projection
def macroprojectins(x,a,b):
    return a*x+b




def npv(cash_flows, wacc, g):
    npv = 0
    for t, cash_flow in enumerate(cash_flows):
        npv += cash_flow / (1 + (wacc/100)) ** t
    return npv

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