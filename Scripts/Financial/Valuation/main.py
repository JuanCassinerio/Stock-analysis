#LIBRERIAS
from datetime import date
import pandas as pd
import plotly.io as pio
from pathlib import Path
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


    ''' cwd = Path.cwd()/'Scripts'/'Financial'/'Valuation'
    financial_statements = pd.read_csv(cwd / 'data.csv')'''
    financial_statements['Date'] = pd.to_datetime(financial_statements['Date'])

    start_date = financial_statements['Date'].iloc[-1]

    end_date = date.today()
    ticker_data={'description': companydescription(ticker), 'financial_statements': financial_statements, 'price': price(ticker, start_date - pd.DateOffset(years=6), end_date)}
    return ticker_data,start_date

def macrodata(start_date):

    SPY = price('SPY',start_date - pd.DateOffset(years=6), date.today())

    rates=fred()['rf'].tail(12*5)
    rf,rf_std=rates.mean(),rates.std()/2

    gdpworld_growth=gdpworld()
    g,g_std=gdpworld_growth['gdp growth'].iloc[-1],gdpworld_growth['gdp growth'].std()/2

    EquityRiskPremium = dmd()[['Year', 'Implied ERP (FCFE)']]['Implied ERP (FCFE)']
    ERP,ERP_std=EquityRiskPremium.mean(),EquityRiskPremium.std()/2

    macros = {'SPY':SPY,'rf':rf,'rf_std':rf_std,'g':g,'g_std':g_std,'ERP':ERP,'ERP_std':ERP_std}

    return macros

def results_plotter(ticker_data,results):

    # Filter the data for the last two years
    ticker_data['price'] = ticker_data['price'][ticker_data['price']['Date'] >= ticker_data['price']['Date'].max() - pd.DateOffset(years=3)]

    fig = make_subplots(rows=2, cols=2, row_heights=[5, 3],column_widths=[5, 1], shared_xaxes=True, vertical_spacing=0.1)

    #price and scenario values

    fig.add_trace(go.Scatter(
        x=ticker_data['price']['Date'],
        y=ticker_data['price']['Adj Close'],
        mode='lines',line=dict(color='blue', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'), row=1, col=1)

    ball_color = ["green", "blue", "red"]

    for i in range(len(results['TarjetPrice_scenarios'])):
        scenario_name = results['TarjetPrice_scenarios']['scenario'].iloc[i]  # Get the scenario name
        target_price = results['TarjetPrice_scenarios']['Tarjet Price'].iloc[i]  # Get the target price

        # Add a scatter plot for each scenario with dots
        fig.add_trace(go.Scatter(
            x=[results['Date_t0']],
            y=[target_price],
            mode='markers',marker=dict(size=20, color=ball_color[i]),name=scenario_name), row=1, col=1)

    y_min = min(ticker_data['price']['Adj Close'].min(), results['TarjetPrice_scenarios']['Tarjet Price'].iloc[2].min())
    y_max = ticker_data['price']['Adj Close'].max()

    fig.update_layout(shapes=[dict(type='line',x0=results['Date_t0'],x1=results['Date_t0']
                                   ,y0=y_min, y1=y_max, line=dict(dash='dash', color='black'))])

    # Revenue Data
    fig.add_trace(go.Scatter(
        x=results['Projected Financial Statements']['Date'],
        y=results['Projected Financial Statements']['Revenue Change']*100,
        mode='lines',line=dict(color='black', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=results['Projected Financial Statements']['Date'],
        y=results['Projected Financial Statements']['Smoothed Revenue Change']*100,
        mode='lines',line=dict(color='dark grey', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'), row=2, col=1)

    company_name = ticker_data['description']['Company']
    industry = ticker_data['description']['industry']
    sector = ticker_data['description']['sector']
    country = ticker_data['description']['country']
    summary = ticker_data['description']['Summary'].split('.')[1]

    def split_text(text, max_length):
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            if sum(len(w) for w in current_line) + len(word) + len(current_line) - 1 <= max_length:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))  # Add the last line
        return '<br>'.join(lines)

    formatted_summary = split_text(summary, 30)

    text_parts_12 = [
        f"<b>Company:</b> {company_name}<br>",
        f"<b>Industry:</b> {industry}<br>",
        f"<b>Sector:</b> {sector}<br>",
        f"<b>Country:</b> {country}<br>",
        f"<b>Summary:</b> {formatted_summary}<br>"]

    text_sizes_12 = [15, 15, 15, 15, 15, 15]
    text_colors_12 = ["black", "black", "black", "black","black", "black"]

    annotations_12 = []
    for i in range(len(text_parts_12)):
        annotations_12.append(dict(text=text_parts_12[i],x=0.77,  y=1 - i * 0.05,  showarrow=False,font=dict(size=text_sizes_12[i], color=text_colors_12[i]),xref="paper", yref="paper",xanchor="left"))

    tp=results['TarjetPrice_scenarios']['Tarjet Price'].iloc[1]
    tp_m=(results['TarjetPrice_scenarios']['Tarjet Price'].iloc[1]-results['TarjetPrice_scenarios']['Tarjet Price'].iloc[1])/2
    p_v = ticker_data['price']['Adj Close'].iloc[(ticker_data['price']['Date'] - results['Date_t0']).abs().idxmin()]

    action = 0
    if tp - tp_m > p_v:
        action = 'Strong Buy'
    elif tp > p_v:
        action = 'Buy'
    elif tp - (tp_m / 2) <= p_v <= tp + (tp_m / 2):
        action = 'Hold'
    elif tp < p_v <= tp + tp_m:
        action = 'Sell'
    else:
        action = 'Strong Sell'


    if results['R2']>60:
        text_parts_22 = [
            f"<b>Intrinsic Value Obtained (DCF):</b>",
            f"<b>Current Price:</b> ${round(ticker_data['price']['Adj Close'].iloc[0],1)}<br>",
            f"<b>Normal Scenario:</b> ${round(results['TarjetPrice_scenarios']['Tarjet Price'].iloc[1],1)}<br>",
            f"<b>Optimistic Scenario:</b> ${round(results['TarjetPrice_scenarios']['Tarjet Price'].iloc[0],1)}<br>",
            f"<b>Pesimistic Scenario:</b> ${round(results['TarjetPrice_scenarios']['Tarjet Price'].iloc[2],1)}<br>",
            f"<b>Status:</b> {action} / {round((tp/p_v -1)*100,1)} %<br>",
        ]
        text_sizes_22 = [20, 15, 15, 15, 15, 20]
        text_colors_22 = ["black", "black", "black", "black","black", "black"]
    else:
        text_parts_22 = [f"<b>No Value. Low Accuracy Result:</b>"]
        text_sizes_22 = [40]
        text_colors_22 = ["black"]

    annotations_22 = []
    for i in range(len(text_parts_22)):
        annotations_22.append(dict(text=text_parts_22[i],x=0.77,y=0.3 - i * 0.05,showarrow=False,font=dict(size=text_sizes_22[i], color=text_colors_22[i]),xref="paper", yref="paper",xanchor="left"))

    annotations_22 = []
    for i in range(len(text_parts_22)):
        annotations_22.append(dict(text=text_parts_22[i], x=0.77, y=0.3 - i * 0.05, showarrow=False,
                                   font=dict(size=text_sizes_22[i], color=text_colors_22[i]), xref="paper",
                                   yref="paper", xanchor="left"))

    annotations_11 = [dict(text="<b>Historic Stock Price [usd]</b>",
        x=0,y=1.05,xref="paper",yref="paper",showarrow=False,  font=dict(size=20, color="black"))]

    annotations_21 = [dict(text="<b>Revenue Growth (Quaterly %)</b>"
        ,x=0,y=0.38,xref="paper",yref="paper",showarrow=False,font=dict(size=20, color="black",) )]

    annotations_title = [dict(text=f"Fundamental Stock Value of <b>{ticker}</b> for last Financial Statement : <b>{results['Date_t0'].strftime('%Y-%m-%d')}</b>",
        x=0,y=1.12,xref="paper",yref="paper",showarrow=False,font=dict(size=30, family="Helvetica", color="black")  )]

    all_annotations = annotations_12 + annotations_22 + annotations_11 + annotations_21 + annotations_title

    fig.update_layout(annotations=all_annotations,showlegend=False)
    fig.show()
    return

if __name__ == "__main__":

    ticker = 'META' # do it with 3 stock of 3 sectors appl nvda(tech) and xom(commodities) amzn(retail) + aapl
    #for ticker in tickerlist(calculate return

    # GET STOCK AND MACROECONOMICAL VARIABLES
    ticker_data,start_date=tickerdata(ticker)
    macros=macrodata(start_date)

    # VALUATE STOCK
    results=damodaran_2(ticker_data,macros)

    # SHOW RESULTS
    results_plotter(ticker_data,results)


    '''
     plot grpah with price taarjet(x2) + yf reference, margin errors, r2and low plot with
    company interest rate is dependant on rf or rfund
    how to project beta
    
    
    ticker_data['financial_statements'].to_csv(cwd/'data.csv')
    
    '''
