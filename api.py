import os
import sys
import json
import requests

import pandas as pd

from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    exit()

symbol = sys.argv[1]

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={os.environ["API_KEY"]}'

r = requests.get(url)

json.dump(
    r.json(),
    open(f'data/raw_data/{symbol.upper()}.json', 'w')
)

data = pd.DataFrame.from_dict(r.json()['Time Series (Daily)'], orient='index').astype(float)

data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

data.to_csv(f'data/csv_data/{symbol.upper()}.csv')