# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:07:21 2023

@author: Kwan
"""

# 옵션가격 불러오기 (매달마다 월물은 만기 2달 전 1일부터 ~ 만기까지 (만기 껴있는 달 20일까지))

import pandas as pd
import numpy as np
from datetime import datetime as dt

df = pd.read_excel("C:/Users/문희관/Desktop/weeklies.xlsx", sheet_name = None, skiprows = 7, index_col = 0)


def sort_loop(df):

    df.columns = df.loc['Name']
    df.drop(df.index[0:5], axis = 0, inplace = True)
    
    # columns to multiindexed columns
    
    dummy = df.columns.drop_duplicates()
    
    callput = []
    exp = df.iat[1, 2]
    exp = dt.strptime(str(exp), "%Y%m%d")
    strike = []
    
    for i in dummy:
        
        ii = i.split(" ")
        
        callput.append(ii[1])
        strike.append(ii[3])
    
    callput = np.unique(callput)
    exp2 = [exp]
    strike = np.unique(strike)
    title = df.loc['D A T E'].unique()
    
    ind = pd.MultiIndex.from_product([callput, exp2, strike, title], names = ['cp' , 'expiry', 'strike', 'title'])
    
    df.drop(index = ['D A T E'], axis = 0, inplace = True)
    df.index.name = None
    
    df2 = df.copy()
    
    df2.columns = ind
    df2.drop("만기일", axis = 1, level = 3, inplace = True)
    df2 = df2.loc[df2.index <= exp]
    
    df_result = df2.melt(ignore_index = False)

    return df_result


dummy_frame = pd.DataFrame()

for keys in df.keys():
            
        result = sort_loop(df[keys])
        
        dummy_frame = pd.concat([dummy_frame, result], axis = 0)
    




