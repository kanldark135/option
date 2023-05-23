#%% 

import pandas as pd
import function as myfunc
import vol_forecast as myvf
import backtest as mybt

# %% 
# 실현변동성 관련 지표

df_daily = pd.read_excel("C:/Users/kanld/Desktop/종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'E:AC').dropna()
df_daily = df_daily.iloc[:, 0:4].sort_index(ascending = True)
df_daily.index.name = 'date'
df_daily.columns = ['open','high','low','close']

df_vkospi = pd.read_excel("C:/Users/kanld/Desktop/종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'A:B').dropna()
df_vkospi = df_vkospi.sort_index(ascending = True)

a = myvf.vol_forecast(df_daily,1)
b = myvf.vol_forecast(df_daily, 5)
c = myvf.vol_forecast(df_daily, 10)
d = myvf.vol_forecast(df_daily, 20)
e = myvf.vol_forecast(df_daily, 30)
f = myvf.vol_forecast(df_daily, 40)

#%% 내재변동성 관련