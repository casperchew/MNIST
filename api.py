import os
import json
import requests

from dotenv import load_dotenv

load_dotenv()

function = 'TIME_SERIES_DAILY'
symbol = 'NVDA'

url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={os.environ["API_KEY"]}'

r = requests.get(url)

json.dump(
    r.json(),
    open(f'{symbol}.json', 'w')
)