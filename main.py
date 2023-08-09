#%% 

import pandas as pd
import numpy as np
import function as myfunc
import vol_forecast as myvf
import backtest as bt
import os

local_user = 'kanld'
path_xls = f"C:/Users/{local_user}/Desktop/"
path = os.getcwd()
path_pkl = path + "/data_pickle/"

# %% 
# 실현변동성 관련 지표 (volscore / 현재 분포에 기반한 임의 추정 / 시계열기반 GARCH 예측 / 머신러닝 기반 예측)

df_daily = pd.read_excel(path_xls + "종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'E:AC').dropna()
df_daily = df_daily.iloc[:, 0:4].sort_index(ascending = True)
df_daily.index.name = 'date'
df_daily.columns = ['open','high','low','close']
df_daily.to_pickle(path_pkl + "/k200.pkl")

df_vkospi = pd.read_excel(path_xls + "종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'A:B').dropna()
df_vkospi = df_vkospi.sort_index(ascending = True)

a = myvf.vol_forecast(df_daily, 1)
b = myvf.vol_forecast(df_daily, 5)
c = myvf.vol_forecast(df_daily, 10)
d = myvf.vol_forecast(df_daily, 20)
e = myvf.vol_forecast(df_daily, 30)
f = myvf.vol_forecast(df_daily, 40)
g = myvf.vol_forecast(df_daily, 50)

table_volscore = pd.DataFrame([a.status()[0], b.status()[0], c.status()[0], d.status()[0], e.status()[0], f.status()[0], g.status()[0]], index = [1, 5, 10, 20, 30, 40, 50])
table_p = pd.DataFrame([a.status()[1], b.status()[1], c.status()[1], d.status()[1], e.status()[1], f.status()[1], g.status()[1]], index = [1, 5, 10, 20, 30, 40, 50])

#%% 내재변동성 관련

monthly = bt.backtest('monthly', 'call')
moneyness_ub = 0
moneyness_lb = 45

atm = 0.14

# 순전히 raw 데이터테이블

datav = monthly.iv_data(moneyness_ub, moneyness_lb)

df_front = datav['front']
df_back = datav['back']
df_fskew = datav['fskew']
df_bskew = datav['bskew']
df_term = datav['term']

# 각 테이블에 대한 descriptive stats

report = monthly.iv_analysis(moneyness_ub, moneyness_lb, atm = atm, quantile = 0.1)
front = report['front'].to_csv("./front.csv")
back = report['back'].to_csv("./back.csv")
fskew = report['fskew'].to_csv("./fskew.csv")
bskew = report['bskew'].to_csv("./bskew.csv")
term = report['term'].to_csv("./term.csv")


# %%
