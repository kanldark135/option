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

# 1) prepping daily data

# Loading the dataset

df_daily = pd.read_excel("C:/Users/kanld/Desktop/rawdata_230421.xlsx", sheet_name = "daily_data", index_col = 0)
df_daily = df_daily.sort_index(ascending = True)

# transform into workable format

daily_close_var = df_daily['close'] ** 2  # 일일 분산
daily_close_vol = np.sqrt(daily_close_var) * np.sqrt(252) # 연율화된 vol

# %% 1. 전체 분포에서 현재 volscore 위치 파악 (probability measure)

volscore_total = df_daily['volscore_total'] # volscore 는 현재는 엑셀파일에서 구하는중이나 언제든지 port 가능
today_volscore = volscore_total.iloc[-1]

volscore_p = func.custom_cdf_function(volscore_total, today_volscore)

# %% 2. volscore 의 현재 추세 파악 (visually, matplot)

weight_vol = np.arange(1, 6)

volscore_trend = volscore_total.tail(120)
volscore_trend_5wma = volscore_total.tail(125).rolling(window = 5).apply(lambda x : np.dot(x, weight_vol)/weight_vol.sum())

fig_daily, axes = plt.subplots()
axes.plot(volscore_trend, label = 'volscore')
axes.plot(volscore_trend_5wma, label = 'volscore_5wma')
axes.legend()

print(volscore_p)
plt.show()

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

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
lin_pred = lin_reg.predict(x_test)

xgb_reg = XGBRegressor()
xgb_reg.fit(x_train, y_train)
xgb_pred = xgb_reg.predict(x_test)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)
rf_pred = rf_reg.predict(x_test)

methods = [lin_reg[1], xgb_reg[1], rf_reg[1]]

print(methods)

#%%    4. 추정된 변동성 분포에서 IV 위치 파악

iv = 0.14

pdf = sstat.gaussian_kde(daily_close_garch['realized_vol'])
xs = np.linspace(0, max(daily_close_garch['realized_vol']), 200)
ys = pdf(xs)

fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.axvline(x = iv, color = 'r')

p = func.custom_cdf_function(daily_close_garch['realized_vol'], 0.14)

print(fig)
print(p)





daily_vol_tr = np.sqrt(df_daily['TR'] ** 2) * np.sqrt(252)
normal_tr = func.garch_process(df_daily['TR'], 7, n_interval = 'day', n_paths = 1000)

# %% 
# weekly anaylsis

df_weekly = pd.read_excel("C:/Users/kanld/Desktop/rawdata_230421.xlsx", sheet_name = "weekly_data", index_col = 0)
df_weekly = df_weekly.sort_index(ascending = True)

weekly_close = func.garch_process(df_weekly['close'], 2, n_interval = 'week', n_paths = 1000)
weekly_tr = func.garch_process(df_weekly['TR'], 2, n_interval = 'week', n_paths = 1000)


