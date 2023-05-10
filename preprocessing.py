import pandas as pd
import numpy as np
import arch

df = pd.read_excel("C:/Users/문희관/Desktop/종합 (2).xlsx", sheet_name = 'data', index_col = 0, usecols = 'E:AC').dropna()
daily_price = df.loc[:, 'open':'close'].sort_index(ascending = True)

# n days cumulative tables - loop

class return_function:

    def __init__(self, df_price):

        self.df_price = df_price
    
    def total_return(self, n_days):
            
        '''close : close_over_close / high : high_over_close / low : low_over_close / minmax : high - low / tr : true return (encompassing minmax + more)'''
            
        high_high = self.df_price.rolling(window = n_days).max()['high']
        low_low = self.df_price.rolling(window = n_days).min()['low']
        last_close = self.df_price['close'].shift(n_days)

        c_over_c = self.df_price['close'] / last_close - 1 # 최종 종가대비 기준 종가
        h_over_c = high_high / last_close - 1 # 기간 내 고가대비 기준 종가
        l_over_c = low_low / last_close - 1  # 기간 내 저가대비 기준 종가
        h_over_l = (high_high - low_low)/last_close  # 기간 내 고가대비 기간 내 저가
        tr = pd.concat([np.abs(h_over_c), 
                        np.abs(l_over_c), 
                        h_over_l], axis = 1).max(axis = 1)  # True return = h_over_c/l_over_c/h_over_l 의 절대값 -> 제일 확대
        
        result = pd.concat([c_over_c, h_over_c, l_over_c, h_over_l, tr], axis = 1)
        result.columns = ['close', 'high', 'low', 'minmax', 'tr']

        return result.dropna()
    
    def plus_return(self, n_days):
            
        plus_only = self.total_return(n_days)
        plus_only = plus_only[plus_only > 0].fillna(0)

        return plus_only.dropna()
    
    def minus_return(self, n_days):
        
        minus_only = self.total_return(n_days)
        minus_only = minus_only[minus_only < 0].fillna(0)

        return minus_only.dropna()

    def intraday_ret(self):
        
        '''h_over_o / l_over_o / c_over_o / h_over_l'''
        
        h_over_o = self.df_price['high'] / self.df_price['open'] - 1 # 시초대비 고가
        l_over_o = self.df_price['low'] / self.df_price['open'] - 1 # 시초대비 저가 
        c_over_o = self.df_price['close'] / self.df_price['open'] - 1 # 시초대비 종가
        h_over_l = self.df_price['high'] / self.df_price['low'] - 1  # 당일고저차

        result = pd.concat([h_over_o, l_over_o, c_over_o, h_over_l], axis = 1)

        return result.dropna()


def volscore(df_return, kwarg = 'close'): ## 상기 close/ high / low/ minmax / tr 컬럼 구조 그대로 따른다는 가정, df_return 은 daily
    ''' kwarg = 'close', 'tr' '''

    df_var = np.power(df_return[kwarg], 2)

    nominal_stdev = np.sqrt(252 * df_var.expanding(1).sum() / df_var.expanding(1).count())
    ma5 = np.sqrt(252 * df_var.rolling(5).sum() / 5)
    ma20 = np.sqrt(252 * df_var.rolling(20).sum() / 20)
    weighted_ma_5 = df_var.rolling(5).apply(lambda var_vector : np.dot(np.arange(1, 6), var_vector))
    weighted_ma_20 = df_var.rolling(20).apply(lambda var_vector : np.dot(np.arange(1, 21), var_vector))
    ewma = df_var.ewm(alpha = 0.06, adjust = False).mean().pipe(lambda x : np.sqrt(x * 252))

    def garch_derivation(df_return):
        df_return = df_return[kwarg] * 100
        model = arch.arch_model(df_return, vol = 'garch', mean = 'zero')
        result = model.fit().params

        return result
    
    coef = garch_derivation(df_return)

    garch = list()

    for i, var in enumerate(df_var.values):
        if i == 0:
            previous_garch_var = var
            garch.append(previous_garch_var)
        else:  
            result = coef[0] + coef[1] * var + coef[2] * garch[i - 1]
            garch.append(result) 