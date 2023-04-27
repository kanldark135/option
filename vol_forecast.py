#%%

import numpy as np
import function as func
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import scipy.stats as sstat
import arch
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

'''
변동성 추정 Process

1. 경험적 수기 조정 과정

오늘까지 실현변동성이 이 수준이고 (=평균 회귀) 
앞으로 오르/내릴 확률이 높아 (=군집성, 현재 vol score 의 추세)

평균회귀계수 : 볼이 평균보다 높거나 / 낮으면 -> 낮아지거나 높아질 가능성 고정값으로 multiplier
추세지속계수 : 볼이 현재 오르고 있거나 / 내리고 있으면 -> 계속 높아지거나 / 낮아질 가능성 고정값 multiplier

-> 2번 추정 모델들의 공통적인 문제로
변동성이 별일 없을때는 비슷하게 움직이나, 
1) spike 발생을 예측 못하는 가운데 후행해버림
2) spike 정도도 과소반영함 (garch 계열의 시계열 모형들이 특히)
3) 이어지는 변동성 축소도 평균회귀계수가 부족하여 후행
그러므로 돌려보면 Garch나 ML이나 사실상 실제 일변동성의 이동평균 느낌으로 연출됨
그러므로 억지로 모멘텀적인 요소 반영하여 오를때는 더오르고 / 내릴때는 더내리는 상황 보정

2) 실제 변동성 안 쓰고 여러 일변동성 평균낸 custom vol score 사용

static 하게 "실제 현재까지 실현변동성" 과 관련된 지표들로 구성된 vol score
해당 vol score 의 분포도에서의 현재 위치 파악 (평균으로 오를 확률 / 내릴 확률)
해당 vol score 의 plot 을 통해 현재 추세 파악 (추세 지속 가능성 파악)

-> 여기서 중요한건 현재 수준이 이정도이므로 앞으로 확대 / 축소될 가능성이 높다! 만 판단하는 것 ("실제 얼마나" 는 밑에 2번에서 추정)

----------------------------------
2. (1번에서 애초에 평균 수준으로 오를것/내릴것이라고 상정해놓고)
만약 오르/내린다면 어떤 식으로 분포를 띄는 가운데 변동성 path 형성할 것인가?

1) stochastic model 에 근거하여 추정한 변동성 -> 다음날 gaussian dist에 투입하여 수익률 도출 -> 해당 수익률을 분산으로 보고 7일간 recursive 구조 형성 -> monte carlo simulation 
2) ML model 에 근거하여 추정한 변동성 -> 다음날 gaussian dist에 투입하여 수익률 산출
-> 해당 수익률을 분산으로 보고 t+1 ML 진행 + 이러한 recursive 구조 7번 형성 -> monte carlo simulation
3) Deep learning 모델로 아예 wholesale 로 7일치 추정

----------------------------------
3. 이러한 분포로 움직인다고 했을때 내 IV가 지금 매력/감당할 수 있는 수준인가?

=> 2번에서 도출된 7일 변동성 추정치들의 분포에 내 IV 대응해서 P 구하기 -> 최종 Weight 비중으로 사용

'''

# %% Daily volatility forecast

# Loading the dataset

df_daily = pd.read_excel("C:/Users/kanld/Desktop/rawdata_230421.xlsx", sheet_name = "daily_data", index_col = 0)
df_daily = df_daily.sort_index(ascending = True)

class vol_forecast:

    label = ['close', 'tr', 'volscore', 'volscore_up_only', "volscore_down_only"]
    
    def __init__(self, df, interval = 'day'):

        self.df = df
        self.df = self.df.sort_index(ascending = True)

        if interval in ['day' , 'week', 'month']:
            self.interval = interval
        else:
            raise ValueError("Input must be day/week/month")

        self.data = dict()
        for i in vol_forecast.label:
            self.data[i] = df[i]

    def step_1(self):

        label = ['volscore', 'volscore_up_only', 'volscore_down_only']
        color = ['g', 'r', 'b']
        self.today_volscore = dict()
        self.volscore_p = dict()

        self.fig_1, axes = plt.subplots(1, 3, figsize = (20, 10))
        bins = np.linspace(0, 1, 200)

        for i, label_i  in enumerate(label):
                df_volscore = self.data.get(label_i)
                today_volscore = df_volscore.iloc[-1]
                volscore_prob = func.custom_cdf_function(df_volscore, today_volscore)
                self.today_volscore[label_i] = today_volscore
                self.volscore_p[label_i] = volscore_prob
                axes[i].hist(df_volscore, density = True, bins = bins, color = color[i])
                axes[i].axvline(x = today_volscore, color = 'black', linewidth = 2)
                axes[i].set_title(label_i)

        return self.today_volscore, self.volscore_p, self.fig_1

    def step_2(self, weight_days = 5):
        
        ''' weight_days : days of averaging'''
        
        label = ['volscore', 'volscore_up_only', 'volscore_down_only']
        weight = np.arange(1, weight_days + 1)


        self.fig_2, axes = plt.subplots(1, len(label), figsize = (40, 5 * len(label)))

        for i in enumerate(label):
            
            trend = self.data.get(i[1]).tail(120)
            trend_wma = self.data.get(i[1]).tail(125).rolling(window = weight_days).apply(lambda x : np.dot(x, weight)/weight.sum())
            axes[i[0]].plot(trend, label = 'trend')
            axes[i[0]].plot(trend_wma, label = 'weighted_trend')
            axes[i[0]].set_title(i[1])
            axes[i[0]].legend()

        return self.fig_2
      
    def forecast_garch(self, iv, n_count, lt_volatility = None, start_date = None, n_paths = 5000):

        ''' 
        Copy of func.garch_process, with=
        variables :
		df_return 은 0.01 = 1% 의 scale 로. "일단위 return" scale 의 데이터만 받음
		n_count 는 generate 하고자 하는 향후 n일의 갯수
		n_interval = 'day' / 'week' / 'month' 만
		lt_volatililty 는 연율화된 표준편차 term으로 작성
		start_date : start_date 의 수익률부터 garch 모형의 표본으로 feed, 없으면 전부

        model_param = res.params, # params of garch model
        today_garch_vol = today_garch_annual, # today's garch volatility 
        forecast_vol = forecast_garch_annual, # one path forecast garch volatility
        forecast_vol_2 = forecast_garch_annual_2, # one path forecast garch volatility v2
        realized_return = cond_return, # one path's realized return over t ~ t+n_count
        realized_vol = realized_return_vol, # one path's aggregate volatility over t ~ t+n_count
        realized_vol_avg = realized_return_vol.mean() # avg of all paths'''
        
        return_label = ['close', 'tr']

        self.vol_result = dict()
        self.prob_result = dict()

        self.fig_3, axes = plt.subplots(1, len(return_label), figsize = (20, 20))

        for i in enumerate(return_label):
    
            df_data = self.df[i[1]]
            garch_vol = func.garch_process(df_data, n_count, n_interval = self.interval, lt_volatility = lt_volatility, start_date = start_date, n_paths = n_paths)['realized_vol']

            self.vol_result[i[0]] = garch_vol, func.custom_cdf_function(garch_vol, iv)
            
            pdf = sstat.gaussian_kde(garch_vol)
            xs = np.linspace(min(garch_vol), max(garch_vol), 200)
            ys = pdf(xs)

            axes[i[0]].plot(xs, ys)
            axes[i[0]].axvline(x = iv, linestyle = '--', color = 'black')
            
        return self.vol_result, self.fig_3

    def forecast_ml_reg(self):
        print("to be written")

    def final_probability(self):
        print('to be written')


# %% 3. 향후 변동성 추정

# 이하 전부 variance 가 아닌 volatility term / 1년 연율화

# 1) model-based prediction : GARCH(1,1)

daily_close_garch = func.garch_process(df_daily['close'], 7, n_interval = 'day', n_paths = 5000)

# 2) ML based vol prediction 향후 추가

# 전처리
# X : close vol, TR vol, volscore_total 3개 -> 평균 0 가정시 표준편차 = abs(수익률)
# Y : 다음날 close vol (close_vol.shift(-1))

predictors = ['close', 'TR', 'volscore_total']
df_x = df_daily[predictors]
df_x.update(np.abs(df_x))
df_y = np.abs(df_daily['close']).shift(-1)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.01, shuffle = False)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_pred = lin_reg.predict(x_test)

xgb_reg = XGBRegressor()
xgb_reg.fit(x_train, y_train)
xgb_pred = xgb_reg.predict(x_test)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)
rf_pred = rf_reg.predict(x_test)


#%%    4. 추정된 변동성 분포에서 IV 위치 파악



daily_vol_tr = np.sqrt(df_daily['TR'] ** 2) * np.sqrt(252)
normal_tr = func.garch_process(df_daily['TR'], 7, n_interval = 'day', n_paths = 1000)

# %% 
# weekly anaylsis

# df_weekly = pd.read_excel("C:/Users/문희관/Desktop/rawdata_230421.xlsx", sheet_name = "weekly_data", index_col = 0)
# df_weekly = df_weekly.sort_index(ascending = True)

# weekly_close = func.garch_process(df_weekly['close'], 2, n_interval = 'week', n_paths = 1000)
# weekly_tr = func.garch_process(df_weekly['TR'], 2, n_interval = 'week', n_paths = 1000)


