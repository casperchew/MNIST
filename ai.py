import json
import time
import numpy as np
import pandas as pd

from tqdm import tqdm

import ANN

res = json.load(open('data/raw_data/NVDA.json'))
data = {}

for i in res['Time Series (Daily)'].keys():
    epoch = time.mktime(time.strptime(i, '%Y-%m-%d'))
    data[epoch] = res['Time Series (Daily)'][i]

data = pd.DataFrame(data).T.astype(float)
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
data.loc[data.index < 1717948800, 'Close'] = data.loc[data.index < 1717948800]['Close'] / 10

model = ANN.ANN(10, 100, 1, lr=1e-1)

x_train = []
y_train = []

for i in range(89):
    x = data.iloc[i:i + 10]
    x_train.append(x['Close'])
    y = data.iloc[i + 11]
    y_train.append(y['Close'])

x_train = np.array(x_train)
y_train = np.array(y_train).reshape(-1, 1)

for _ in tqdm(range(1000000)):
    model.train(x_train, y_train)
model.save('NVDA')

print(model(x_train))
# print(model(x_train[2].reshape(1, -1)))