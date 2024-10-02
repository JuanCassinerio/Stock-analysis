'''
- Cedear list ARG


'''


import requests
from io import BytesIO
import pandas as pd


def comafi():


    url = "https://www.comafi.com.ar/Multimedios/otros/7279.xlsx?v=11092024"

    # Respect robots.txt and terms of service
    response = requests.head(url)
    df=[]
    if response.status_code == 200:
        # Check robots.txt for any restrictions on downloading
        robots_url = f"https://www.comafi.com.ar/robots.txt"
        robots_response = requests.get(robots_url)

        if robots_response.status_code == 200:
            # Check robots.txt for disallowing downloads
            if not any(
                    line.lower().startswith("disallow: /Multimedios/") for line in robots_response.text.splitlines()):
                # Proceed with download if robots.txt allows
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()  # Raise exception for non-200 status codes

                    # Download data in chunks to avoid memory issues for large files
                    data = BytesIO()
                    for chunk in response.iter_content(1024):
                        data.write(chunk)

                    # Load data as DataFrame
                    df = pd.read_excel(data, engine='openpyxl')  # Use openpyxl for better compatibility
                    df=df.dropna()

                    df = df.rename(columns={
                        'Unnamed: 0': 'Index',
                        'Unnamed: 1': 'Company Name',
                        'Unnamed: 2': 'ticker',
                        'Unnamed: 3': 'ISIN',
                        'Unnamed: 4': 'CUSIP',
                        'Unnamed: 5': 'SEDOL',
                        'Unnamed: 6': 'Ratio',
                        'Unnamed: 7': 'Multiplier',
                        'Unnamed: 8': 'Quantity'
                    })
                    df=df[['Company Name','ticker','Multiplier']]


                    print("Data successfully downloaded and converted to DataFrame.")
                except requests.exceptions.RequestException as e:
                    print(f"Download failed: {e}")
            else:
                print("Downloading the file is disallowed by robots.txt.")
        else:
            print(f"Failed to access robots.txt: {robots_response.status_code}")
    else:
        print(f"Failed to check the URL: {response.status_code}")
    return df