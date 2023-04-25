# %% 

# 연환산 표준편차로 preprocessing

import pandas as pd
import scipy.stats as sstat
import numpy as np
import scipy.optimize as sopt
import arch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# 연환산 표준편차로 preprocessing

df_daily = pd.read_excel("C:/Users/문희관/Desktop/rawdata_230421.xlsx", sheet_name = 'daily_data', index_col = 0, header = 0)
df_daily = df_daily.sort_index(ascending = True)

x = ['close', 'TR', 'volscore_total']

df_data = df_daily[x]
df_data.update(df_data[['close', 'TR']].apply(lambda x : np.sqrt(252 * x ** 2)))

df_data = df_data.assign(y = df_data['close'].shift(-1))

n_days = 6

use_last_n_data = 1000 # use only awhen planning to train sliding window
train_data = 980

model = XGBRegressor()

## could rely on train_test_split() method but not precise enough to split n number of tests

predicted_return = []
predicted_vol = []

for i in range(n_days):
    df_x = df_data.iloc[-use_last_n_data:][x]
    df_y = df_data.iloc[-use_last_n_data:]['y']

    x_train = df_x[:train_data]
    x_test = df_x[train_data:]

    y_train = df_y[:train_data]
    y_test = df_y[train_data:]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

# validation
mse = mean_squared_error(y_test.iloc[:-1].values, y_pred[:-1])
print(mse)

plt.plot(y_test.iloc[:-1].values, label = 'actual')
plt.plot(y_pred, label = 'pred')
plt.legend()
