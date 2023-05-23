# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:36:21 2023

@author: kanld
"""
# %% preprocessing

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# local database 에 접근해서 가격 불러오기

monthly_path = "C:/Users/문희관/Desktop/option/data_pickle/monthly"
weekly_path = "C:/Users/문희관/Desktop/option/data_pickle/weekly"
kospi_path = "C:/Users/문희관/Desktop/option/data_pickle/주가"

df_monthly = pd.read_pickle(monthly_path)
df_weekly = pd.read_pickle(weekly_path)
df_kospi = pd.read_pickle(kospi_path)

def find_closest_strike(x, interval = 2.5):
    divided = divmod(x, interval)
    if divided[1] >= 1.25: 
        result = divided[0] * interval + interval
    else:
        result = divided[0] * interval
    return result

find_closest_strike = np.vectorize(find_closest_strike)

# 만기당일은 종가 X, 행사가격으로 계산

def value_at_exp(callput, s, k):
    if callput == "c":
        result = max(s - k, 0)
    else:
        result = max(k - s, 0)
    return result

# 근/차/그 외 월물 각각 labeling

atm = df_kospi.assign(atm = find_closest_strike(df_kospi['close']))['atm']

# 1) ATM 대비 Moneyness 표기

df_c = df_monthly[df_monthly['cp']== "C"]
df_c = df_c.merge(atm, how = 'inner', left_index = True, right_on = atm.index)
df_c = df_c.assign(moneyness = df_c.strike - df_c.atm)

df_p = df_monthly[df_monthly['cp']== "P"]
df_p = df_p.merge(atm, how = 'inner', left_index = True, right_on = atm.index)
df_p = df_p.assign(moneyness = df_p.atm - df_p.strike)

# 2) 매 날짜별로 grouping 후 group 내에서 근/차물 식별하여 레이블링

def grouping(df):
    df['cycle'] = 0
    df_grouped = df.groupby(by = df.index)

    keys = df_grouped.groups.keys()

    dummy_array = list()

    for i in keys:
        part_df = df_grouped.get_group(i)
        exp_list = part_df.expiry.unique()

        for j in part_df.itertuples():
            if j.expiry == exp_list[0]:
                dummy_array.append('front')
            elif j.expiry == exp_list[1]:
                dummy_array.append("back")
            elif j.expiry == exp_list[2]:
                dummy_array.append("backback")
            else:
                dummy_array.append("farther")
        
    df['cycle'] = dummy_array
    
    return df

df_c = df_c.pipe(grouping).drop(columns = ['key_0'])
df_p = df_p.pipe(grouping).drop(columns = ['key_0'])

# 콜
df_cv = df_c[df_c['title'] == '내재변동성']
df_cp = df_c[df_c['title'] == '가격']

# 풋
df_pv = df_p[df_p['title'] == '내재변동성']
df_pp = df_p[df_p['title'] == '가격']

# %% 변동성 분석 툴

import seaborn as sns

# call

class call_vol:

    def __init__(self, df_cv):

        self.df_civ = df_cv[(df_cv['moneyness'] > -10) & (df_cv['moneyness']< 50)]  # 엥간치 필요할만한 구간만 추리기

        self.df_front_civ = self.df_civ[self.df_civ['cycle'] == 'front'] # 근월물
        self.df_back_civ = self.df_civ[self.df_civ['cycle'] == 'back'] # 차월물

    def _to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def civ_front_index(self, moneyness_lb, moneyness_ub, remove_dte = [1]):

        remove_dte = self._to_list(remove_dte)
         
        cond = {'cond1' : self.df_front_civ['moneyness'] >= moneyness_lb, 
                'cond2' : self.df_front_civ['moneyness'] < moneyness_ub,
                'cond3' : ~self.df_front_civ['dte'].isin(remove_dte)
                }
        front_index = self.df_front_civ[
        (cond['cond1'] & cond['cond2'] & cond['cond3'])
        ]
        result = front_index.groupby(by = front_index.index)['value'].mean()
        return result
    
    def civ_back_index(self, moneyness_lb, moneyness_ub, remove_dte = [1]):

        remove_dte = self._to_list(remove_dte)
    
        cond = {'cond1' : self.df_back_civ['moneyness'] >= moneyness_lb, 
                'cond2' : self.df_back_civ['moneyness'] < moneyness_ub,
                'cond3' : ~self.df_back_civ['dte'].isin(remove_dte)
                }
        back_index = self.df_back_civ[
        ((cond['cond1']) & (cond['cond2']) & (cond['cond3']))
        ]
    
        result = back_index.groupby(by = back_index.index)['value'].mean()
        return result
    
    def civ_specific(self, moneyness, dte):
        
        # vectorize variables to fit them in isin(), although they are scalars

        moneyness = self._to_list(moneyness)
        dte = self._to_list(dte)

        cond = {'cond_1' : self.df_civ['moneyness'].isin(moneyness),
                'cond_2' : self.df_civ['dte'].isin(dte)
        }

        result = self.df_civ[(cond['cond_1']) & (cond['cond_2'])]
        fig, ax = plt.subplots(1, 1)
        
        sns.catplot(data = result, x = 'moneyness', y = 'value', hue = result.index, ax = ax)

        return result, fig
    
    ## 1) 동일만기내 skewness
    # 2) 근-차월물간 IV 스프레드의 scoring 내지는 차트

    tentative_merged = pd.merge(a.df_front_civ, a.df_back_civ, how = 'left', left_on = [a.df_front_civ.index, 'strike'], right_on = [a.df_back_civ.index, 'strike'])

        
    

# skewness 데이터

# 


# %%

# 이미 식별된 42DTE table

# dte_42 = ['2008-01-04', '2008-02-01', '2008-02-29', '2008-03-28', '2008-05-02', '2008-05-30', '2008-07-04', '2008-08-01', '2008-08-29', '2008-10-02', '2008-10-31', '2008-11-28', '2009-01-02', '2009-01-30', '2009-02-27', '2009-04-03', '2009-04-30', '2009-05-29', '2009-07-03', '2009-07-31', '2009-08-28', '2009-10-01', '2009-10-30', '2009-12-04', '2009-12-29', '2010-01-29', '2010-02-26', '2010-04-02', '2010-04-30', '2010-05-28', '2010-07-02', '2010-07-30', '2010-09-03', '2010-10-01', '2010-10-29', '2010-12-03', '2010-12-30', '2011-01-28', '2011-03-04', '2011-04-01', '2011-04-29', '2011-06-03', '2011-07-01', '2011-07-29', '2011-09-02', '2011-09-30', '2011-10-28', '2011-12-02', '2011-12-29', '2012-01-27', '2012-03-02', '2012-03-30', '2012-05-04', '2012-06-01', '2012-06-29', '2012-08-03', '2012-08-31', '2012-09-28', '2012-11-02', '2012-11-30', '2013-01-04', '2013-02-01', '2013-02-28', '2013-03-29', '2013-05-03', '2013-05-31', '2013-06-28', '2013-08-02', '2013-08-30', '2013-10-04', '2013-11-01', '2013-11-29', '2014-01-03', '2014-01-28', '2014-02-28', '2014-03-28', '2014-05-02', '2014-05-30', '2014-07-04', '2014-08-01', '2014-08-28', '2014-10-02', '2014-10-31', '2014-11-28', '2015-01-02', '2015-01-30', '2015-02-27', '2015-04-03', '2015-04-30', '2015-05-29', '2015-07-03', '2015-07-31', '2015-08-28', '2015-10-02', '2015-10-30', '2015-12-04', '2015-12-29', '2016-01-29', '2016-03-04', '2016-04-01', '2016-04-29', '2016-06-03', '2016-07-01', '2016-07-29', '2016-09-02', '2016-09-30', '2016-10-28', '2016-12-02', '2016-12-29', '2017-01-26', '2017-03-03', '2017-03-31', '2017-04-28', '2017-06-02', '2017-06-30', '2017-08-04', '2017-09-01', '2017-09-29', '2017-11-03', '2017-12-01', '2017-12-28', '2018-01-26', '2018-03-02', '2018-03-30', '2018-05-04', '2018-06-01', '2018-06-29', '2018-08-03', '2018-08-31', '2018-09-28', '2018-11-02', '2018-11-30', '2019-01-04', '2019-02-01', '2019-02-28', '2019-03-29', '2019-05-03', '2019-05-31', '2019-06-28', '2019-08-01', '2019-08-30', '2019-10-04', '2019-11-01', '2019-11-29', '2020-01-03', '2020-01-31', '2020-02-28', '2020-04-03', '2020-04-29', '2020-05-29', '2020-07-03', '2020-07-31', '2020-08-28', '2020-09-18', '2020-10-30', '2020-12-04', '2020-12-30', '2021-01-29', '2021-02-26', '2021-04-02', '2021-04-30', '2021-05-28', '2021-07-02', '2021-07-30', '2021-09-03', '2021-10-01', '2021-10-29', '2021-12-03', '2021-12-30', '2022-01-28', '2022-03-04', '2022-04-01', '2022-04-29', '2022-06-03', '2022-07-01', '2022-07-29', '2022-09-02', '2022-09-30', '2022-10-28', '2022-12-02']
# dte_42 = pd.to_datetime(dte_42)

# # 실제 dte 42에 근접한 옵션의 만기만 추려내기

# dummy_42 = df_monthly.loc[dte_42]
# dummy_42 = dummy_42[(dummy_42['dte'] > 40) & (dummy_42['dte'] < 46)] 

# # dte_42에 해당되는 날짜에 살아있는 옵션들이 근월/차월물 여러개 있을텐데 실제 DTE=42일 내지는 42일에 준하는 놈만 추려야 함

# expiry_table_42 = dummy_42['expiry'].drop_duplicates()


# # 이미 식별된 21DTE table

# dte_21 = ['2008-01-25', '2008-02-22', '2008-03-21', '2008-04-18', '2008-05-23', '2008-06-20', '2008-07-25', '2008-08-22', '2008-09-19', '2008-10-24', '2008-11-21', '2008-12-19', '2009-01-23', '2009-02-20', '2009-03-20', '2009-04-24', '2009-05-22', '2009-06-19', '2009-07-24', '2009-08-21', '2009-09-18', '2009-10-23', '2009-11-20', '2009-12-24', '2010-01-22', '2010-02-19', '2010-03-19', '2010-04-23', '2010-05-20', '2010-06-18', '2010-07-23', '2010-08-20', '2010-09-24', '2010-10-22', '2010-11-19', '2010-12-24', '2011-01-21', '2011-02-18', '2011-03-25', '2011-04-22', '2011-05-20', '2011-06-24', '2011-07-22', '2011-08-19', '2011-09-23', '2011-10-21', '2011-11-18', '2011-12-23', '2012-01-20', '2012-02-17', '2012-03-23', '2012-04-20', '2012-05-25', '2012-06-22', '2012-07-20', '2012-08-24', '2012-09-21', '2012-10-19', '2012-11-23', '2012-12-21', '2013-01-25', '2013-02-22', '2013-03-22', '2013-04-19', '2013-05-24', '2013-06-21', '2013-07-19', '2013-08-23', '2013-09-17', '2013-10-25', '2013-11-22', '2013-12-20', '2014-01-24', '2014-02-21', '2014-03-21', '2014-04-18', '2014-05-23', '2014-06-20', '2014-07-25', '2014-08-22', '2014-09-18', '2014-10-24', '2014-11-21', '2014-12-19', '2015-01-23', '2015-02-17', '2015-03-20', '2015-04-24', '2015-05-22', '2015-06-19', '2015-07-24', '2015-08-21', '2015-09-18', '2015-10-23', '2015-11-20', '2015-12-24', '2016-01-22', '2016-02-19', '2016-03-25', '2016-04-22', '2016-05-20', '2016-06-24', '2016-07-22', '2016-08-19', '2016-09-23', '2016-10-21', '2016-11-18', '2016-12-23', '2017-01-20', '2017-02-17', '2017-03-24', '2017-04-21', '2017-05-19', '2017-06-23', '2017-07-21', '2017-08-25', '2017-09-22', '2017-10-20', '2017-11-24', '2017-12-22', '2018-01-19', '2018-02-14', '2018-03-23', '2018-04-20', '2018-05-25', '2018-06-22', '2018-07-20', '2018-08-24', '2018-09-21', '2018-10-19', '2018-11-23', '2018-12-21', '2019-01-25', '2019-02-22', '2019-03-22', '2019-04-19', '2019-05-24', '2019-06-21', '2019-07-19', '2019-08-22', '2019-09-20', '2019-10-25', '2019-11-22', '2019-12-20', '2020-01-23', '2020-02-21', '2020-03-20', '2020-04-24', '2020-05-22', '2020-06-19', '2020-07-24', '2020-08-21', '2020-09-18', '2020-10-20', '2020-11-20', '2020-12-24', '2021-01-21', '2021-02-19', '2021-03-19', '2021-04-23', '2021-05-21', '2021-06-18', '2021-07-23', '2021-08-20', '2021-09-24', '2021-10-22', '2021-11-19', '2021-12-24', '2022-01-21', '2022-02-18', '2022-03-25', '2022-04-22', '2022-05-20', '2022-06-24', '2022-07-22', '2022-08-19', '2022-09-23', '2022-10-21', '2022-11-18', '2022-12-23']
# dte_21 = pd.to_datetime(dte_21)

# dummy_21 = df_monthly.loc[dte_21]
# dummy_21 = dummy_21[(dummy_21['dte'] > 18) & (dummy_21['dte'] < 25)]

# expiry_table_21 = dummy_21['expiry'].drop_duplicates()


# # 7DTE table for weeklies
# # 해결해야할 문제 1) 8DTE 가격은 전주물 만기랑 날짜 겹쳐서 그룹핑 궁리 해야 함
# # 7DTE (금요일) 종가 변동성 통상적으로 주말꺼 미리 빼서 구림

# expiry_table_7 = df_weekly[df_weekly['dte'] == 7]['expiry'].drop_duplicates()


# # 일반적인 N dte 에서의 날짜 - 월물 table * 주의) dte가 휴무 등으로 없으면 해당 월물은 backtest 안됨

# n_dte = 35
# expiry_table_n = df_monthly[df_monthly['dte'] == n_dte]['expiry'].drop_duplicates()
