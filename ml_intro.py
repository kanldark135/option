
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# preprocessing steps

#1 name predictors and target

df_daily = pd.read_excel("C:/Users/kanld/Desktop/rawdata_230421.xlsx", sheet_name = "daily_data", header = 0, index_col = 0)
df_daily.sort_index(ascending = True, inplace = True)

data = df_daily

predictors = ['close', 'TR', 'volscore_total']
df_x = data[predictors]
df_x.update(np.abs(df_x['close']))

df_y = (np.abs(data['close'])).shift(-1)

#2. (if dataframe then convert to numpy array)

#2 train test split

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.01, shuffle = False)

#3 scale

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#4. model fit

xgb_reg = XGBRegressor()
xgb_reg.fit(x_train, y_train)

#5. predict on x_test

y_pred = xgb_reg.predict(x_test)

#6. compare the result with y_test

y_pred = pd.DataFrame(y_pred, index = y_test.index)

fig, ax = plt.subplots()
ax.plot(y_pred, label = 'pred')
ax.plot(y_test, label = 'values')
ax.legend()

#7. compute the robustness of your analysis with either mse or mape

mse = mean_squared_error(y_test.iloc[:-1], y_pred.iloc[:-1])
print(mse)

# 5~7 can be integrated into loop structure

