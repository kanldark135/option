#%% 
import pandas as pd
import numpy as np
import function as myfunc
import matplotlib.pyplot as plt
import calc
import seaborn as sns

# %% 데이터 추출 클래스

class backtest:

    ''' period = monthly / weekly, callput = 'call / put'''

    def __init__(self, monthlyorweekly = 'monthly', callput = 'put'):

        route = f"./data_pickle/{callput}_{monthlyorweekly}.pkl"
        self.df = pd.read_pickle(route) # 전체

        self.df_front = self.df[self.df['cycle'] == 0] # 근월물
        self.df_back = self.df[self.df['cycle'] == 1] # 차월물
        self.df_backback = self.df[self.df['cycle'] == 2] # 차차월물

    def _to_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]
        
    def matrix(self, component = 'iv', frontback = 'front', moneyness_lb = 0, moneyness_ub = 30, remove_dte = [1, 2, 3, 4, 5, 6, 7]):
        
        '''components : iv, price, delta, gamma, theta, vega'''

        remove_dte = self._to_list(remove_dte)

        if frontback == 'front':
            df = self.df_front
        elif frontback == 'back':
            df = self.df_back
        else:
            df = self.df
        
        cond = {'cond1' : df['moneyness'] >= moneyness_lb, 
                'cond2' : df['moneyness'] <= moneyness_ub,
                'cond3' : ~df['dte'].isin(remove_dte)
                }
        df_cond = df[
        (cond['cond1'] & cond['cond2'] & cond['cond3'])
        ]

        bool_unique = ~df_cond.index.duplicated()

        res = df_cond.pivot_table(values = component, index = df_cond.index, columns = 'moneyness')
        res = res.merge(df_cond[['dte', 'close']].loc[bool_unique], how = 'left', left_on = res.index , right_index = True)
        res['atm'] = res[0]

        return res
    
    def iv_data(self, moneyness_lb = 0, moneyness_ub = 30, remove_dte = [1,2,3,4,5,6,7]):

        remove_dte = self._to_list(remove_dte)
    
    # 1) raw iv data

        df_front = self.matrix(component = 'iv', frontback = 'front', moneyness_lb = moneyness_lb, moneyness_ub = moneyness_ub, remove_dte = remove_dte)
        df_back = self.matrix(component = 'iv', frontback = 'back', moneyness_lb = moneyness_lb, moneyness_ub = moneyness_ub, remove_dte = remove_dte)

        cut_front = df_front.loc[:, moneyness_lb : moneyness_ub]
        cut_back = df_back.loc[:, moneyness_lb : moneyness_ub]

    # 2) skewness over atm

        front_skew = cut_front.pipe(np.divide, cut_front[[0]])\
            .interpolate(methods = 'polynomial', order = 2, axis = 1)\
            .combine_first(df_front)
        back_skew = cut_back.pipe(np.divide, df_back[[0]])\
            .interpolate(methods = 'polynomial', order = 2, axis = 1)\
            .combine_first(df_back)

    # 3) term spread, dte 는 front 기준
    # back 에 있는 0 때문에 발생하는 문제 (저 0은 데이터 정리할때 nan 값 안받는 함수때문에 어쩔수없이 nan -> 0 으로 바꾼거임)
        
        front_over_back = cut_front.divide(cut_back)
        front_over_back = front_over_back \
            .dropna(how = 'all') \
            .replace(np.inf, np.nan) \
            .interpolate(methods = 'polynomial', order = 2, axis = 1) \
            .merge(df_front[['atm', 'dte', 'close']], how = 'left', left_index = True, right_index = True)

        res = {
            'front' : df_front,
            'back' : df_back, 
            'fskew' : front_skew,
            'bskew' : back_skew,
            'term' : front_over_back
            }
        
        return res
    
    def iv_analysis(self, moneyness_lb = 0, moneyness_ub = 30, quantile = 0.5, atm = None, dte = None, remove_dte = [1,2,3,4,5,6,7]):

        '''유사 atm IV 수준만 골라서 비교하려면 atm 에 값 기록'''
        ''''''
        data = self.iv_data(moneyness_lb, moneyness_ub, remove_dte = remove_dte)

        res = {}

        # 1) descriptive stats

        for key in data.keys():

            df = data.get(key)
            # n/a interpolated 2nd polynomial
            df.update(df.loc[:, moneyness_lb:moneyness_ub].interpolate(methods = 'polynomial', order = 2, axis = 1))
            
            #basic stats

            stats = df.describe()

            def similar_atm(df, atm, buffer = 0.1):
                cond_1 = df['atm'] > atm - buffer
                cond_2 = df['atm'] < atm + buffer
                df_similar_atm = df[cond_1 & cond_2]
                
                return df_similar_atm
            
            if atm != None:
                df = similar_atm(df, atm)
            else:
                pass
        
            dummy = df.apply(lambda x : np.quantile(x, quantile)).to_frame(quantile).T
            stats = pd.concat([stats, dummy])

            res[key] = stats

        return res

# %%

import backtest as bt

a = bt.backtest()

data = a.iv_data()

front = data['front']
back = data['back']
fskew = data['fskew']
bskew = data['bskew']
term = data['term']

report = a.iv_analysis(quantile = 0.1)

# %%
