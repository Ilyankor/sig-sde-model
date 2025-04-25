import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sklearn
import sklearn.model_selection
from src.util.brownian import estimate_brownian
import esig
from tqdm import tqdm

# get the data
data_dir = Path("data/stocks.db")
sql_query = """
    SELECT date, close
    FROM daily_tech
    WHERE symbol = 'aapl' AND date > '2017-01-01'
    """

with sqlite3.connect(data_dir) as conn:
    cur = conn.cursor()
    cur.execute(sql_query)
    rows = cur.fetchall()

days = [datetime.strptime(x[0],'%Y-%m-%d') for x in rows]
close = np.array([x[1] for x in rows])

# set window size
window = 1001

# split the data
split = sklearn.model_selection.train_test_split(np.arange(len(rows)-window), train_size=0.7)
train_indices = split[0]
test_indices = split[1]

# define the model
lasso_reg = 1e-5
model = sklearn.linear_model.SGDRegressor(
    penalty="l1",
    alpha=lasso_reg,
    max_iter=10000,
    tol=0.0001,
    learning_rate="adaptive"
)

# signature params
channels = 2
depth = 3

# since the signature always starts with 1, ignore it
sig_keys = esig.sigkeys(channels, depth).strip().split(' ')
features = len(sig_keys) - 1

t = np.linspace(0.0, 1.0, num=window-1)
for i in tqdm(train_indices):
    x = close[i:i+window-1]           # select data
    x = x / x[0]                    # normalize to first entry
    brown = estimate_brownian(x)    # estimate brownian motion
    y = close[i+1:i+window]       # data result
    y = y / y[0]

    brownian_data = np.column_stack((t, brown))

    # compute signature
    data = np.zeros((window-1, features))

    for i in range(2, window):
        data[i-1, :] = esig.stream2sig(brownian_data[:i, :], depth)[1:]

    # fit the model
    model.partial_fit(data, y)

plt.plot(x)
plt.plot(y)
plt.show()

# j = test_indices[0]
# X = close[j:j+window-1]
# Y = close[j+1:j+window]
# Y = Y/Y[0]
# X = X/X[0] 
# brown = estimate_brownian(X)
# brownian_data = np.column_stack((t, brown))
# data = np.zeros((window-1, features))

# for i in range(2, window):
#     data[i-1, :] = esig.stream2sig(brownian_data[:i, :], depth)[1:]
# guess = model.predict(data)

# plt.plot(t, Y, label="Real price")
# plt.plot(t, guess, label="Predicted price")
# plt.legend()
# plt.grid()
# plt.show()
