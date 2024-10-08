'''
new_directory = 'C:/Users/Usuario/Desktop/repos/Scripts/Financial/dark box'  # Replace with the path to your desired directory
os.chdir(new_directory)
'''

#LIBRERIAS
from datetime import date
import pandas as pd
import plotly.io as pio
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'

#MODULES
from Scripts.Financial.Database.ticker_scrape import price,companydescription,av_financials
from Scripts.Financial.Valuation.valuation import damodaran_2
from Scripts.Financial.Database.macro_scrape import dmd,gdpworld,fred

# FUNCIONES
def tickerdata(ticker):
    key = 'B6T9Z1KKTBKA2I1C'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    financial_statements = av_financials(ticker, key, headers) #alphavantage 2009-today quaterly frecuency

    #fin = financialdata(ticker) #yahoo last 4 years
    start_date = financial_statements['Date'].iloc[-1]


    end_date = date.today()
    ticker_data={'description': companydescription(ticker), 'financial_statements': financial_statements, 'price': price(ticker, start_date - pd.DateOffset(years=6), end_date)}
    return ticker_data,start_date

def macrodata(start_date):

    SPY = price('SPY',start_date - pd.DateOffset(years=6), date.today())

    rates=fred()['rf'].tail(12*5).mean()
    rf,rf_std=rates.mean(),rates.std()/2

    gdpworld_growth=gdpworld()
    g,g_std=gdpworld_growth['gdp growth'].iloc[-1],gdpworld_growth['gdp growth'].std()/2

    EquityRiskPremium = dmd()[['Year', 'Implied ERP (FCFE)']]['Implied ERP (FCFE)']
    ERP,ERP_std=EquityRiskPremium.mean(),EquityRiskPremium.std()/2

    macros = {'SPY':SPY,'rf':rf,'rf_std':rf_std,'g':g,'g_std':g_std,'ERP':ERP,'ERP_std':ERP_std}

    return macros

def results_plotter(ticker_data,results,save_path):

    results['Date_t0']
    results['R2']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ym, mode='markers', name='fitting'))
    fig.add_trace(go.Scatter(x=ex['Date'], y=ex['gdp growth'], mode='markers', name='extrapolation data'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='present data'))
    fig.update_layout(width=700, height=400)
    fig.show()


    # Create subplots with the specified height ratios
    fig = make_subplots(rows=2, cols=2, row_heights=[5, 1], shared_xaxes=True, vertical_spacing=0.1)

    # Example of adding data to the subplots
    # Plot something in the first subplot
    fig.add_trace(go.Trace(x=[1, 2, 3], y=[4, 5, 6], mode='lines', name='Line 1'), row=1, col=1)
    fig.add_trace(go.Trace(x=[1, 2, 3], y=[4, 5, 6], mode='lines', name='Line 1'), row=1, col=1)


    fig.add_trace(go.Scatter(
        x=ticker_data['price']['Date'],
        y=ticker_data['price']['Adj Close'],
        mode='lines',
        line=dict(color='blue', width=2),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ticker_data['price']['Date'],
        y=ticker_data['price']['Adj Close'],
        mode='lines',
        line=dict(color='blue', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=2, col=1)

    #descrption
    fig.add_trace(go.Scatter(
        x=ticker_data['price']['Date'],
        y=ticker_data['price']['Adj Close'],
        mode='lines',
        line=dict(color='blue', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=2, col=1)

    #results
    fig.add_trace(go.Scatter(
        x=ticker_data['price']['Date'],
        y=ticker_data['price']['Adj Close'],
        mode='lines',
        line=dict(color='blue', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=2, col=1)

    fig.update_layout(height=600, width=800, title_text="Two Subplots with Plotly", showlegend=True)

    # Show the plot
    fig.show()


    for i in range(0,len(results['TarjetPrice_scenarios'])):
        results['TarjetPrice_scenarios']['Tarjet Price'].iloc[i]
        results['TarjetPrice_scenarios']['scenario']





    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [5, 1]})



    ax1.text(0.5, 0.95,
             f"Número de quiebres VaR = {len(breaks)} en {len(merged_df)} / {round(len(breaks) / len(merged_df) * 100, 2)}%",
             horizontalalignment='center', transform=ax1.transAxes, fontsize=14)

    ax1.legend()
    ax1.legend(fontsize=12)
    ax1.set_ylabel('USD')
    ax1.tick_params(axis='x', rotation=45, which='both', bottom=False, labelbottom=False)
    ax1.set_title(f"Agregacion: {report['SubAgg']}", fontsize=20, fontweight='bold')
    ticker_data['description']['Company']
    ticker_data['description']['industry']
    ticker_data['description']['sector']
    ticker_data['description']['country']
    ticker_data['description']['Summary'].split('.')[1]
    link_yf = ticker_info['link_yf']

    target_mean_price = ticker_data['description']['targetMeanPrice']
    target_high_price = ticker_data['description']['targetHighPrice']
    target_low_price = ticker_data['description']['targetLowPrice']

    ax2.plot(results['Projected Financial Statements']['Date'],results['Projected Financial Statements']['Revenue Change']
             ,color='black', linestyle='-', linewidth=1, zorder=2)
    ax2.plot(results['Projected Financial Statements']['Date'], results['Projected Financial Statements']['Smoothed Revenue Change']
             ,color = 'black', linestyle = '--', linewidth = 1, zorder = 2)
    ax2.set_ylabel('Revenue Growth Rate', color='black')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(folder_path / f'grafica {graphname}.png')

    return

'''
    if r2<60 discard result, or red alert and others 
    ("low accuracy result"
 
     "model prediction not converngent instead of results on reuslts box")
'''



if __name__ == "__main__":

    cwd = Path.cwd()

    ticker = 'AAPL' # do it with 3 stock of 3 sectors appl nvda(tech) and xom(commodities) amzn(retail) + aapl
    #for ticker in tickerlist(calculate return

    # GET STOCK AND MACROECONOMICAL VARIABLES
    ticker_data,start_date=tickerdata(ticker)
    macros=macrodata(start_date)

    # VALUATE STOCK
    results=damodaran_2(ticker_data,macros,cwd)

    # SHOW RESULTS
    #results_plotter(ticker_data,results)


    '''
     plot grpah with price taarjet(x2) + yf reference, margin errors, r2and low plot with
    company interest rate is dependant on rf or rfund
    how to project beta
    
    '''