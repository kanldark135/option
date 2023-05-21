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

def mc_garch(daily_return, n_days, start_date = 0, n_paths = 10000):

	start_date = pd.to_datetime(start_date)
	
	model = arch.arch_model(100 * daily_return, vol = 'garch', mean = 'zero')
	fit = model.fit(first_obs = start_date, disp = "off")

	omega = fit.params.loc['omega']
	alpha = fit.params.loc['alpha[1]']
	beta = fit.params.loc['beta[1]']
	lt_mean = np.sqrt(252 * omega / (1 - alpha - beta) / 10000)

	ret = np.zeros((n_days+1, n_paths))
	ret[0] = daily_return[-1]

	var_pred = np.zeros((n_days+1, n_paths))
	var_pred[0] = (fit.conditional_volatility[-1] ** 2) / 10000

	for i in range(n_days):

		pred = omega / 10000 + alpha * ret[i] ** 2 + beta * var_pred[i]
		var_pred[i+1] = pred
		ret_pred = np.random.normal(0, np.sqrt(var_pred[i]))
		ret[i+1] = ret_pred

	var_pred = np.sqrt(252 * var_pred)
	
	return var_pred[n_days], {'params' : fit.params, 'lt_mean' : lt_mean}


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