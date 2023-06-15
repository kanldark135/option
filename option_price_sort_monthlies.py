# -*- coding: utf-8 -*-
"""
@author: Kwan
"""

# 옵션가격 불러오기 (매달마다 월물은 만기 2달 전 1일부터 ~ 만기까지 (만기 껴있는 달 20일까지))

import pandas as pd
import numpy as np
from datetime import datetime as dt


pre = pd.read_pickle("C:/Users/문희관/Desktop/option/data_pickle/monthly.pkl")

df = pd.read_excel("C:/Users/문희관/Desktop/옵션데이터/2023.xlsx", skiprows = 7, index_col = 0)

# 불필요한 데이터 정리
df.columns = df.loc['Name']
df.index.name = 'date'
df.drop(df.index[0:5], axis = 0, inplace = True)

# 종목 - 만기매칭 Series 따로 생성
bool_idx = df.loc['D A T E'].isin(['만기일'])
df_exp = df.iloc[3].loc[bool_idx].apply(lambda x : dt.strptime(str(x), "%Y%m%d"))

# callput / expiry / strike / 속성값으로 구성된 multiindex 생성
callput = list()
expiry = list()
strike = list()

for i in df.columns:
    expiry.append(df_exp[i])
    ii = i.split(" ")
    callput.append(str(ii[1]))
    strike.append(float(ii[3]))

title = df.loc['D A T E'] # 종가 / 내재변동성 두개로 구성된 값
expiry = pd.to_datetime(expiry)

ind = pd.MultiIndex.from_arrays([callput, expiry, strike, title], names = ['cp', 'expiry', 'strike','title'])

df2 = df.copy()
df2.columns = ind

idx = pd.IndexSlice
df2 = df2.loc[:, idx[:, :, :, ~bool_idx]] # 만기일 column 은 전부 제거
df2.drop(index = ['D A T E'], axis = 0, inplace = True) # D A T E 행도 전부 제거
df2.index = pd.to_datetime(df2.index)

df_result = df2.melt(ignore_index = False)
df_result = df_result.assign(dte = (df_result.expiry - df_result.index) / pd.Timedelta(1, "D") + 1)
df_result = df_result.loc[df_result['dte'] >= 1]

# 0제거
post = df_result[df_result.value != 0]

# 둘이 합치기
final = pd.concat([pre, post], axis = 0)
final = final.drop_duplicates()

# pickle 파일로 다시 빼기
final.to_pickle("C:/Users/문희관/Desktop/option/data_pickle/monthly.pkl")

