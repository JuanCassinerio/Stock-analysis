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
    ticker_data['price'] = ticker_data['price'][
        ticker_data['price']['Date'] >= ticker_data['price']['Date'].max() - pd.DateOffset(years=2)]

    # Create subplots with the specified height ratios
    fig = make_subplots(rows=2, cols=2, row_heights=[5, 1],column_widths=[5, 2], shared_xaxes=True, vertical_spacing=0.1)


    #results
    fig.add_trace(go.Scatter(
        x=ticker_data['price']['Date'],
        y=ticker_data['price']['Adj Close'],
        mode='lines',
        line=dict(color='blue', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=1, col=1)

    for i in range(len(results['TarjetPrice_scenarios'])):
        scenario_name = results['TarjetPrice_scenarios']['scenario'].iloc[i]  # Get the scenario name
        target_price = results['TarjetPrice_scenarios']['Tarjet Price'].iloc[i]  # Get the target price

        # Add a scatter plot for each scenario with dots
        fig.add_trace(go.Scatter(
            x=[results['Date_t0']],  # Assuming 'Date_t0' is a single date or array with matching size
            y=[target_price],
            mode='markers',  # Show as dots
            marker=dict(size=10),  # Marker size
            name=scenario_name  # Scenario name for each trace
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results['Projected Financial Statements']['Date'],
        y=results['Projected Financial Statements']['Revenue Change'],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=results['Projected Financial Statements']['Date'],
        y=results['Projected Financial Statements']['Smoothed Revenue Change'],
        mode='lines',
        line=dict(color='dark grey', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=2, col=1)

    company_name = ticker_data['description']['Company']
    industry = ticker_data['description']['industry']
    sector = ticker_data['description']['sector']
    country = ticker_data['description']['country']
    summary = ticker_data['description']['Summary'].split('.')[1]
    link_yf = ticker_data['description']['link_yf']

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

    # Split the summary into multiple lines (e.g., 80 characters max per line)
    formatted_summary = split_text(summary, 60)

    text_parts = [
        f"<b>Company:</b> {company_name}<br>",
        f"<b>Industry:</b> {industry}<br>",
        f"<b>Sector:</b> {sector}<br>",
        f"<b>Country:</b> {country}<br>",
        f"<b>Summary:</b> {formatted_summary}<br>",  # Use the formatted summary with line breaks
        f"<a href='{link_yf}'>Yahoo Finance</a>"
    ]

    # Define text sizes and colors
    text_sizes = [40, 12, 12, 12, 12, 10]
    text_colors = ["black", "gray", "gray", "gray", "gray", "blue"]

    # Create annotations with defined sizes and colors
    annotations_12 = []
    for i in range(len(text_parts)):
        annotations_12.append(
            dict(
                text=text_parts[i],
                x=1,  # Position at the right edge (relative to subplot width)
                y=1 - i * 150,  # Adjust vertical spacing as needed
                showarrow=False,
                font=dict(size=text_sizes[i], color=text_colors[i]),
                xref="x2",  # Reference x-axis 2
                yref="y1"  # Reference y-axis 1
            )
        )



    annotations_22 = []
    for i in range(len(text_parts)):
        annotations_22.append(
            dict(
                text=text_parts[i],
                x=1,  # Position at the right edge (relative to subplot width)
                y=1 - i * 10,  # Adjust vertical spacing as needed
                showarrow=False,
                font=dict(size=text_sizes[i], color=text_colors[i]),
                xref="x2",  # Reference x-axis 2
                yref="y2"  # Reference y-axis 1
            )
        )

    all_annotations = annotations_12 + annotations_22  # Combine both lists of annotations

    # Update layout to include the text in subplot (row 1, col 2)
    fig.update_layout(
        xaxis2_anchor="x1",  # Anchor x-axis 2 to x-axis 1 for alignment
        yaxis1_anchor="y2",  # Anchor y-axis 1 to y-axis 2 for alignment
        annotations=all_annotations
    )

    # Show the figure
    fig.show()

    results['R2']


    ax1.text(0.5, 0.95,
             f"Número de quiebres VaR = {len(breaks)} en {len(merged_df)} / {round(len(breaks) / len(merged_df) * 100, 2)}%",
             horizontalalignment='center', transform=ax1.transAxes, fontsize=14)

    fig.add_trace(go.Scatter(
        x=results['Projected Financial Statements']['Date'],
        y=results['Projected Financial Statements']['Smoothed Revenue Change'],
        mode='lines',
        line=dict(color='dark grey', width=2, dash='dash'),  # Line color and width
        name='Historic Stock Price'  # Label for the trace
    ), row=2, col=1)

    ticker_data['description']['Company']
    ticker_data['description']['industry']
    ticker_data['description']['sector']
    ticker_data['description']['country']
    ticker_data['description']['Summary'].split('.')[1]
    link_yf = ticker_info['link_yf']

    target_mean_price = ticker_data['description']['targetMeanPrice']
    target_high_price = ticker_data['description']['targetHighPrice']
    target_low_price = ticker_data['description']['targetLowPrice']


    # Show the plot
    fig.show()

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