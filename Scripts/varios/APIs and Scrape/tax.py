#libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import date


folder_path0 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/companiesmarketcap.csv'
folder_path1 = 'C:/Users/Usuario/Desktop/Scripts/Base de datos/tax.csv'

companies = pd.read_csv(folder_path0)
tickers =companies['Symbol'].iloc[:500].tolist()


tax_data = []
i=0
for ticker in tickers:
    i=i+1
    
    print(i,ticker)
    url = f"https://valueinvesting.io/{ticker}/valuation/wacc"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    
    td_elements = soup.find_all('td', string='Tax rate')
    if td_elements:  # Check if any matching elements were found
        tr_element = td_elements[0].find_parent('tr').get_text()
        tax_rate_text = tr_element.split('<td>Tax rate</td>')[-1].split('</tr>')[0].strip()
        
        # Check if tax_rate_text is equal to "Tax rate"
        if tax_rate_text == "Tax rate-" or not any(char.isdigit() for char in tax_rate_text):
            tax_data.append({'Ticker': ticker, 'Tax Rate': '-'})
            continue  # Skip processing for this ticker and move to the next one
        
        # Extract tax rate values and calculate the average
        tax_rate_values = re.findall(r'\d+\.\d+|\d+', tax_rate_text)
        tax_rate = (float(tax_rate_values[0]) + float(tax_rate_values[1])) / 2
        
        # Append the ticker and tax rate to the list
        tax_data.append({'Ticker': ticker, 'Tax Rate': tax_rate})
    else:
        # Skip this ticker if no tax rate information found
        tax_data.append({'Ticker': ticker, 'Tax Rate': '-'})
        continue

# Create a DataFrame from the collected data
df = pd.DataFrame(tax_data)
df['Date'] = date.today()
df.to_csv(folder_path1, index=False)


############################### mETHOD 2 Effective Tax Rate




url = f"https://www.discoverci.com/companies/AAPL/effective-tax-rate#:~:text=Apple%20Inc%20(AAPL)%20Effective%20Tax,quarter%20ended%20December%2030th%2C%202023."
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

graph_element = soup.find("div", class_="graph-container")  # Replace with specific class or tag



if graph_element:
    # You found the graph element! Analyze its attributes or nested elements for further insights.
    print("Found graph element:", graph_element)
else:
    print("Couldn't find a clear graph element using this approach.")











