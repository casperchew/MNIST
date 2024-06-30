import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    exit()

symbol = sys.argv[1]

data = pd.read_csv(f'data/csv_data/{symbol}.csv', index_col=0)

if symbol.upper() == 'NVDA':
    stock_split = data.index.get_loc('2024-06-06')

data.index = np.linspace(1, data.shape[0], num=data.shape[0])

if symbol.upper() == 'NVDA':
    data.loc[stock_split:, 'Close'] = data.loc[stock_split:, 'Close'] / 10

plt.grid()
plt.plot(data.index, data['Close'][::-1], '.-k')
plt.ylim(bottom=0)
plt.show()