import json
import time
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print = pprint

res = json.load(open('NVDA.json'))
data = {}

for i in res['Time Series (Daily)'].keys():
    epoch = time.mktime(time.strptime(i, '%Y-%m-%d'))
    data[epoch] = res['Time Series (Daily)'][i]

json.dump(data, open('data.json', 'w'))
data = pd.DataFrame(data).T.astype(float)
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
data.loc[data.index < 1717948800, 'Close'] = data.loc[data.index < 1717948800]['Close'] / 10
data.to_csv('NVDA.csv')

plt.plot(data.index, data['Close'])
plt.show()