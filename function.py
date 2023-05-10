# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 02:07:44 2023

@author: kanld
"""

import pandas as pd
import scipy.stats as sstat
import numpy as np
import scipy.optimize as sopt
import arch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def gbm_process(spot, drift, vol, n_days, n_paths = 1):
	
	dt = 1/365   
	growth_path = np.exp((drift - vol **2 /2) * dt + vol * np.random.normal(0, np.sqrt(dt), size = (n_days, n_paths)))
	growth_path = np.cumprod(growth_path, axis = 0)
	price_path = spot * growth_path
		
	return price_path
		
def garch_process(df_return, n_count, n_interval, lt_volatility = None, start_date = None, n_paths = 1):

	''' variables :
		df_return 은 0.01 = 1% 의 scale 로
		n_count 는 generate 하고자 하는 향후 n일의 갯수
		n_interval = 'day' / 'week' / 'month' 만
		lt_volatililty 는 연율화된 표준편차 term으로 작성
		start_date : start_date 의 수익률부터 garch 모형의 표본으로 feed, 없으면 전부

		stochastic garch volatility process = 
		V_t+1 = V_t + gamma * (long_term_V - V_t) * dt + alpha * sqrt(2) * V_t * N(0, dt**2)
		
		return:
        0: model_param = res.params, # params of garch model
        1: today_garch_vol = today_garch_annual, # today's garch volatility 
        2: forecast_vol = forecast_garch_annual, # one path forecast garch volatility
        3: forecast_vol_2 = forecast_garch_annual_2, # one path forecast garch volatility v2
        4: realized_return = cond_return, # one path's realized return over t ~ t+n_count
        5: realized_vol = realized_return_vol, # one path's aggregate volatility over t ~ t+n_count
        6: realized_vol_avg = realized_return_vol.mean() # avg of all paths'''

	interval_dict = dict(day = 252, week = 52, month = 12)

	if n_interval in interval_dict.keys():
		n_interval = interval_dict[n_interval]
	else:
		raise ValueError("n_interval input is inappropriate")
	
	# 1) parameter estimation using arch library and 100x scaled return data
	
	arch_return = arch.arch_model(100 * df_return, vol = 'garch', mean = 'Zero')

	if start_date == None:
		res = arch_return.fit()

	else:
		start_date = pd.Timestamp(start_date)
		res = arch_return.fit(first_obs = start_date)
	
	omega = res.params.loc['omega']
	alpha = res.params.loc['alpha[1]'] # weight of latest variance at t
	beta = res.params.loc['beta[1]'] # weight of cumulative beta weighted average variance up to t-1
	gamma = 1 - alpha - beta # speed / weight of mean reversion to lt_var

	if lt_volatility == None:
		lt_var = omega / (1 - alpha - beta) / 10000 ## long_term average variance over corresponding interval (eg) daily var if daily returns)
	
	else:
		lt_var = lt_volatility ** 2 / n_interval

	today_return = df_return.iloc[-1]
	today_return_var = today_return ** 2
	today_garch_var = (res.conditional_volatility.iloc[-1]/100) ** 2
	today_predicted_var = gamma * lt_var + alpha * today_return_var + beta * today_garch_var # predicted today for tomorrow variance

	# 2) generating multiple conditional return and variance paths

	cond_return = np.zeros((n_count, n_paths))
	cond_return[0] = today_return

	cond_var = np.zeros((n_count, n_paths))
	cond_var[0] = today_predicted_var

	cond_var_2 = np.zeros((n_count, n_paths))
	cond_var_2[0] = today_predicted_var

	for i in range(1, n_count): ### 1) deterministic garch model 안에서 stochastic 주가 process + 전날에 추정된 garch_predicted vol 사용

		tomorrow_return = np.random.normal(0, np.sqrt(cond_var[i-1]))
		tomorrow_predicted_var = gamma * lt_var + alpha * (tomorrow_return ** 2) + beta * cond_var[i-1] 
		cond_return[i], cond_var[i] = tomorrow_return, tomorrow_predicted_var

	# for i in range(1, n_count): ### 2) volatility stochastic process 로 아예 치환 (= 사실상 garch 에서 return process 를 stochastic 으로 상정하여 항 정리만 한 셈)

	# 	tomorrow_predicted_var_2 = cond_var_2[i-1] + gamma * (lt_var - cond_var_2[i-1]) + alpha * np.sqrt(2) * cond_var_2[i-1] * np.random.normal(0, 1)
	# 	cond_var_2[i] = tomorrow_predicted_var_2

	# 3) annualizing for better view

	today_garch_annual = res.conditional_volatility.iloc[-1] * np.sqrt(n_interval) / 100
	forecast_garch_annual = np.sqrt(cond_var * n_interval)
	# forecast_garch_annual_2 = np.sqrt(cond_var_2 * n_interval)
	realized_return_vol = np.sqrt(n_interval * sum(np.power(cond_return, 2)) / n_count)

	result = dict(model_param = res.params, # params of garch model
	today_garch_vol = today_garch_annual, # today's garch volatility 
	forecast_vol = forecast_garch_annual, # one path forecast garch volatility
	# forecast_vol_2 = forecast_garch_annual_2, # one path forecast garch volatility v2
	realized_return = cond_return, # one path's realized return over t ~ t+n_count
	realized_vol = realized_return_vol, # one path's aggregate volatility over t ~ t+n_count
	realized_vol_avg = realized_return_vol.mean() # avg of all paths
	)

	return result

def custom_cdf_function(daily_close_vol, target_x, start_x = None):

	'''평균 분산 등과 같은 parameter 없이 data 그 자체에서 pdf 도출'''

	pdf = sstat.gaussian_kde(daily_close_vol)
	
	if start_x == None:
		compute_p = pdf.integrate_box(-np.inf, target_x)
	else:
		compute_p = pdf.integrate_box(start_x, target_x)
		
	return compute_p

# xs = np.linspace(min(daily_close_vol), max(daily_close_vol), 1000)
# ys = pdf(xs)
# fig, ax = plt.subplots()
# ax.plot(xs, ys)
# return fig

def reg_predict(train, test, predictors, target, model, n_estimators = 100, random_state = 0):
	
	''' You must have completed preprocessing dataset into train, test with relavant fit/format
	1) divide dataset into train, test(validation)

	manually generate x_train, x_test, y_train, y_test or 
	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = %)
	
	2) scale, fit and transform
	
	either manually fit and transform outside the function or
	use sklearn.preprocessing.StandardScaler/.... => scaler = StandardScaler() and scaler.fit_transform(x_train / x_test)'''
	
	model = model()

	model.fit(train[predictors], train[target])

	pred = model.predict(test[predictors])

	pred = pd.DataFrame(pred, index = test.index)
	pred = pd.concat([pred, test[target]], axis = 1)
	pred.columns = ['pred', 'actual']
	
	mse = mean_squared_error(pred['actual'][:-1], pred['pred'][:-1])
	
	return pred, mse

def recursive_ml(df_x, df_y, reg_model, n_days, n_paths = 1, test_size = None):

	''' 바로 전날 데이터까지 학습'''
	''' predicted value 는 그래서 바로 다음날꺼 하나'''

	x_train = np.array(df_x.iloc[:-1])
	x_test = np.array(df_x.iloc[-1]).reshape(1, -1)
	y_train = np.array(df_y.iloc[:-1])
	y_test = df_y.iloc[-1] # autoregression 에서 어짜피 y_test는 NA임

	scalar = StandardScaler()
	x_train = scalar.fit_transform(x_train)
	x_test = scalar.transform(x_test)

	# x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size, shuffle = False)

	model_reg = reg_model()
	
	ret = []
	vol = []

	for i in n_days:
		
		model_reg.fit(x_train, y_train)
		model_pred = model_reg.predict(x_test)

		pred_vol = model_pred[-1]

		tomorrow_return = np.random.normal(0, pred_vol)
		ret.append(tomorrow_return)
		vol.append(np.abs(tomorrow_return))




def discrete_kelly(p_vector, outcome_vector):
	
	p_vector = np.array(p_vector)
	outcome_vector = np.array(outcome_vector)
	
	# sum of log expected outcomes
	
	def inverted_log_return(kelly_ratio):	
		result = - np.sum(p_vector * np.log(1 + kelly_ratio * outcome_vector))	
		return result

	ratio = sopt.minimize(inverted_log_return, x0 = 0.5).x
	expected_return = -sopt.minimize(inverted_log_return, x0 = 0.5).fun
	
	return ratio, expected_return


def continuous_kelly(mean, stdev):
	
	# 2 order taylor approximation
	
	symmetrical_dist = mean / (mean ** 2 + stdev ** 2)
	
	# 3 order
	
	skewness = 0 ## compute scistat.skew(dataset) from a given sample
	lambda_3 = stdev**3 * skewness + 3 * mean * stdev **2 - mean ** 3
	asymmetrical_dist = symmetrical_dist + (lambda_3 * mean ** 2) / ((mean **2 + stdev **2) **2)
	
	return symmetrical_dist, asymmetrical_dist


def fraction_of_kelly(kelly_ratio, mean, stdev):
	
	growth_rate = (kelly_ratio - np.power(kelly_ratio, 2) / 2) * (mean ** 2 / stdev ** 2)
	
	return growth_rate