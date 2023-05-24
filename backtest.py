# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:36:21 2023

@author: kanld
"""
# %% preprocessing

import pandas as pd
import numpy as np
import function as myfunc
import matplotlib.pyplot as plt

# local database 에 접근해서 가격 불러오기

def preprocessing(df_option, df_kospi):

    '''return order by df_cv, df_cp, df_pv, df_pp'''

    def find_closest_strike(x, interval = 2.5):
        divided = divmod(x, interval)
        if divided[1] >= 1.25: 
            result = divided[0] * interval + interval
        else:
            result = divided[0] * interval
        return result

    find_closest_strike = np.vectorize(find_closest_strike)

    # 쪼갤수 있는 dimension : 콜/풋, 행사가, 만기 => 콜/풋만 따로 df 만들고 행사가 만기는 같은 df 에서 처리

    # 1) df_kospi join

    df_append = df_kospi.assign(atm = find_closest_strike(df_kospi['close']))

    # 2) ATM 대비 Moneyness 표기

    df_c = df_option[df_option['cp']== "C"]
    df_c = df_c.merge(df_append, how = 'inner', left_index = True, right_on = df_append.index)
    df_c = df_c.assign(moneyness = df_c.strike - df_c.atm)

    df_p = df_option[df_option['cp']== "P"]
    df_p = df_p.merge(df_append, how = 'inner', left_index = True, right_on = df_append.index)
    df_p = df_p.assign(moneyness = df_p.atm - df_p.strike)

    # 3) 매일마다 존재하는 옵션 근/차월물 식별하여 레이블링

    def grouping(df):
        df['cycle'] = 0
        grouped = df.groupby(by = df.index)
        keys = grouped.groups.keys()
        dummy_array = list()

        for i in keys:
            part_df = grouped.get_group(i)
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

    # 4) 최종적으로 콜가격/콜변동성/풋가격/풋변동성 4개의 dataframe 생성

    df_cv = df_c[df_c['title'] == '내재변동성']
    df_cp = df_c[df_c['title'] == '종가']

    # 만기당일은 종가 -> 행사가 Payoff 로 교체
    df_cp['value'] = df_cp['value'].mask(df_cp.index == df_cp['expiry'], np.maximum(df_cp['close'] - df_cp['strike'], 0))

    # 풋
    df_pv = df_p[df_p['title'] == '내재변동성']
    df_pp = df_p[df_p['title'] == '종가']
    df_pp['value'] = df_pp['value'].mask(df_pp.index == df_pp['expiry'], np.maximum(df_pp['strike'] - df_pp['close'], 0))

    return df_cv, df_cp, df_pv, df_pp

# %% 변동성 분석 클래스

import seaborn as sns
from sklearn.linear_model import LinearRegression

class vol_backtest:

    def __init__(self, df_v):

        self.df_iv = df_v[(df_v['moneyness'] >= -10) & (df_v['moneyness'] <= 50)]  # 엥간치 필요할만한 구간만 추리기

        self.df_front_iv = self.df_iv[self.df_iv['cycle'] == 'front'] # 근월물
        self.df_back_iv = self.df_iv[self.df_iv['cycle'] == 'back'] # 차월물

    def _to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def iv_index(self, frontback = 'front', moneyness_lb = 0, moneyness_ub = 15, remove_dte = 1):

        ''' 매일마다 lb~ub 까지의 내재변동성을 평균내서 점수화 => 별도의 모듈에서 PDF 그려서 확률 등등 구하기'''

        remove_dte = self._to_list(remove_dte)

        if frontback == 'front':
            df = self.df_front_iv
        elif frontback == 'back':
            df = self.df_back_iv
        else:
            df = self.df_iv
        
        cond = {'cond1' : df['moneyness'] >= moneyness_lb, 
                'cond2' : df['moneyness'] <= moneyness_ub,
                'cond3' : ~df['dte'].isin(remove_dte)
                }
        df_cond = df[
        (cond['cond1'] & cond['cond2'] & cond['cond3'])
        ]
        result = df_cond.groupby(by = df_cond.index)['value'].mean().dropna()

        return result
    
    def iv_specific(self, moneyness, dte):

        ''' 특정 Moneyness / 특정 DTE 시점에서 과거에 동일한 시점에는 어땠는지 골라서 보기 '''

        moneyness = self._to_list(moneyness)
        dte = self._to_list(dte)

        cond = {'cond_1' : self.df_iv['moneyness'].isin(moneyness),
                'cond_2' : self.df_iv['dte'].isin(dte)
        }
        result = self.df_iv[(cond['cond_1']) & (cond['cond_2'])]

        fig, axes = plt.subplots()
        fig = sns.swarmplot(data = result, x = 'moneyness', y = 'value', ax = axes)

        return result, fig
    
    def iv_skew(self, frontback = 'front', moneyness_lb = 0, moneyness_ub = 20, remove_dte = 1):

        ''' lb ~ ub 까지의 내재변동성의 Skewness 를 지수화해서 보기. 지수화는 각각 linreg, 행사가별 IV의 차이의 합, 비율의 합, 비율의 곱연산'''

        if frontback == 'front':
            df = self.df_front_iv
        elif frontback == 'back':
            df = self.df_back_iv
        else:
            df = self.df_iv
        
        remove_dte = self._to_list(remove_dte)

        cond = {'cond1' : (df['moneyness'] >= moneyness_lb),
                'cond2' : (df['moneyness'] <= moneyness_ub),
                'cond3' : (~df['dte'].isin(remove_dte))}
        df_cond = df[cond.get('cond1') & cond.get('cond2') & cond.get('cond3')]

        grouped = df_cond.groupby(by = df_cond.index)
        
        # 일마다 lb ~ ub IV 의 회귀계수 구하는 nested function

        def linreg(df):
            X = df.moneyness.fillna(0).values
            Y = 100 * df.value.fillna(0).values
            reg = LinearRegression()
            reg.fit(X.reshape(-1, 1), Y)

            return reg.coef_
        
        reg = dict()
        dif_sum = dict()
        pct_sum = dict()
        pct_prod = dict()

        for date, i in grouped:
            i = i.sort_values('moneyness')
            reg[date] = i.pipe(linreg) # 일마다 회귀계수
            dif_sum[date] = (i.value - i.shift(1).value).sum() # 일마다 IV차이의 합
            pct_sum[date] = i.value.pct_change(1).sum() # 일마다 IV 비율의 합
            pct_prod[date] = (i.value.pct_change(1) + 1).prod()  #일마다 (IV 비율 + 1) 의 곱연산
            
        result = pd.DataFrame([reg, dif_sum, pct_sum, pct_prod]).T
        result.columns = ['reg', 'dif_sum', 'pct_sum', 'pct_prod']
        result = result.applymap(float)

        return result

    def iv_calendar(self, moneyness_lb = 0, moneyness_ub = 20, remove_dte = 1):

        ''' 근-차월물간 IV 차이를 차원축소하여 scoring / 각각 뺄셈 후 전부 더하기/평균/근차월간 비율의 전부 더하기/평균'''

        remove_dte = self._to_list(remove_dte)
        
        df_front = self.df_front_iv[(self.df_front_iv['moneyness'] >= moneyness_lb) & (self.df_front_iv['moneyness'] <= moneyness_ub) & (~self.df_front_iv['dte'].isin(remove_dte))]
        df_back = self.df_back_iv[(self.df_back_iv['moneyness'] >= moneyness_lb) & (self.df_back_iv['moneyness'] <= moneyness_ub) & (~self.df_back_iv['dte'].isin(remove_dte))]

        df_total = df_front.merge(df_back['value'], how = 'left', 
                                           left_on = [df_front.index, df_front['strike']],
                                           right_on = [df_back.index, df_back['strike']]
        ).set_index('key_0').rename(columns = {'value_x' : 'front_iv', 'value_y' : 'back_iv'})
        
        grouped = df_total.groupby(by = df_total.index)

        def temp_subtract(df):
            res = df['front_iv'].sub(df['back_iv'], axis = 'index')
            return res
        
        # 1/2. one on one / front from back 뺄셈 후 전부 더하기 / 뺄셈 후 평균
        dif_sum = dict()
        dif_mean = dict()

        # 3/4. one on one / front over back 비율 구해서 전부 더하기 / 비율 평균
        ratio_sum = dict()
        ratio_mean = dict()

        # 5. 위에 근차월 iv_index 의 차이로 갈음해도...?

        for date, i in grouped:
            i = i.sort_values('moneyness')
            dif_sum[date] = i.pipe(temp_subtract).sum()
            dif_mean[date] = i.pipe(temp_subtract).mean()
            ratio_sum[date] = (i['front_iv'] / i['back_iv'] - 1).sum()
            ratio_mean[date] = (i['front_iv'] / i['back_iv']).mean()

        result = pd.DataFrame.from_dict([dif_sum, dif_mean, ratio_sum, ratio_mean]).T
        result.columns = ['dif_sum', 'dif_mean', 'ratio_sum', 'ratio_mean']
        result = result.applymap(float)

        return result

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
