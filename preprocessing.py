import pandas as pd
import numpy as np
import arch

'''일수익률만 있어도 날짜조정해서 n일 누적수익률 구할 수 있음'''

df = pd.read_excel("C:/Users/kanld/Desktop/종합.xlsx", sheet_name = 'data', index_col = 0, usecols = 'E:AC').dropna()
df_daily = df.iloc[:, 0:4].sort_index(ascending = True)
df_daily.index.name = 'date'
df_daily.columns = ['open','high','low','close']


# n days cumulative tables - loop

class return_function:

    ''' feed any price dataset with OHLC format as df_price'''

    def __init__(self, df_price):

        self.df_price = df_price
    
    def total_return(self, n_days):
            
        '''close : close_over_close / high : high_over_close / low : low_over_close / minmax : high - low / tr : true return (encompassing minmax + more)'''
            
        high_high = self.df_price.rolling(window = n_days).max()['high']
        low_low = self.df_price.rolling(window = n_days).min()['low']
        last_close = self.df_price['close'].shift(n_days)

        c_over_c = self.df_price['close'] / last_close - 1 # 1) 최종 종가대비 기준 종가
        h_over_c = high_high / last_close - 1 # 2) 기간 내 고가대비 기준 종가
        l_over_c = low_low / last_close - 1  # 3) 기간 내 저가대비 기준 종가
        h_over_l = (high_high - low_low) / last_close - 1  # 기간 내 고가-저가대비 기준 종가
        tr = pd.concat([np.abs(h_over_c), 
                        np.abs(l_over_c), 
                        h_over_l], axis = 1).max(axis = 1)  
                        # 4) True return = max(h_over_c/l_over_c/h_over_l) 의 절대값 -> 제일 확대
        
        result = pd.concat([c_over_c, h_over_c, l_over_c, tr], axis = 1)
        result.columns = ['close', 'high', 'low', 'tr']

        return result.dropna()
    
    def plus_return(self, n_days):
            
        plus_only = self.total_return(n_days)
        plus_only = plus_only[plus_only > 0].fillna(0)

        return plus_only[['close', 'high']].dropna()
    
    def minus_return(self, n_days):
        
        minus_only = self.total_return(n_days)
        minus_only = minus_only[minus_only < 0].fillna(0)

        return minus_only[['close', 'low']].dropna()

    def intraday_ret(self):
        
        '''h_over_o / l_over_o / c_over_o / h_over_l'''
        
        h_over_o = self.df_price['high'] / self.df_price['open'] - 1 # 시초대비 고가
        l_over_o = self.df_price['low'] / self.df_price['open'] - 1 # 시초대비 저가 
        c_over_o = self.df_price['close'] / self.df_price['open'] - 1 # 시초대비 종가
        h_over_l = self.df_price['high'] / self.df_price['low'] - 1  # 당일고저차

        result = pd.concat([h_over_o, l_over_o, c_over_o, h_over_l], axis = 1)

        return result.dropna()


def volscore(df_return, price = 'close', n = 252): ## 상기 close/ high / low / tr 컬럼 구조 그대로 따른다는 가정, df_return 은 daily
    ''' kwarg = 'close', 'high', ' low, 'tr' '''
    ''' n = annualizing factor = must be chosen according to df_return interval'''

    if price not in ['close', 'high', 'low', 'tr']:

        raise ValueError("Price must be in close / high / low / tr")

    else:

        df_var = np.power(df_return[price], 2)

        daily = np.sqrt(n * df_var) # 1일 변동성
        ma5 = np.sqrt(n * df_var.rolling(5).sum() / 5).fillna(0) # 5일 변동성
        ma20 = np.sqrt(n * df_var.rolling(20).sum() / 20).fillna(0) # 20일 변동성
        wma_5 = np.sqrt(n * df_var.rolling(5).apply(lambda var_vector : np.dot(np.arange(1, 6), var_vector)) / np.arange(1, 6).sum()).fillna(0)
        wma_20 = np.sqrt(n * df_var.rolling(20).apply(lambda var_vector : np.dot(np.arange(1, 21), var_vector)) / np.arange(1, 21).sum()).fillna(0)
        ewma = df_var.ewm(alpha = 0.06, adjust = False).mean().pipe(lambda x : np.sqrt(x * n)).fillna(0)

        # garch derivation
        model = arch.arch_model(df_return[price] * 100, vol = 'garch', mean = 'zero')
        fit = model.fit(disp = "off")
        garch = np.sqrt(n) * fit.conditional_volatility / 100

        result = (daily + ma5 + ma20 + wma_5 + wma_20 + ewma + garch) / 7

    return result.dropna()
