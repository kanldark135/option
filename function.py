# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 02:07:44 2023

@author: kanld
"""

import pandas as pd
import scipy.stats as scistat
import numpy as np
import scipy.optimize as sciop


def gbm_process(spot, drift, vol, n_days, n_paths = 1):
    
    dt = 1/365   
    growth_path = np.exp((drift - vol **2 /2) * dt + vol * scistat.norm.rvs(0, np.sqrt(dt), size = [n_days, n_paths]))
    growth_path = np.cumprod(growth_path, axis = 0)
    price_path = spot * growth_path
        

    return price_path
        
def garch_process(today_return, lt_vol, n_days, n_paths = 1):

    ''' lt_vol input in annual term'''

    ''' stochastic garch volatility process = 
        V_t+1 = V_t + omega * (long_term_V - V_t) * dt + alpha * sqrt(2) * V_t * N(0, dt**2)'''
    ''' omega / alpha / beta derived elsewhere in GARCH parameter estimation'''

    today_variance = today_return ** 2
    lt_variance = lt_vol ** 2 / 252

    today_variance = today_return ** 2
    lt_variance = lt_vol ** 2 / 252

    alpha = 0.06
    beta = 0.90
    omega = 1 - alpha - beta # 0.02
    
    cond_return = np.zeros(n_days)
    cond_var = np.zeros(n_days)
    cond_return[0] = today_return
    cond_var[0] = today_variance

    cond_var_2 = np.zeros(n_days)
    cond_var_2[0] = today_variance

    ## 차후 할 것 : loop 구조 -> numpy 구조로 변경

    for i in range(1, n_days): ### 주가에 대한 stochasticity 유지하면서 변동성은 따라만 가는

        tomorrow_return = np.random.normal(0, np.sqrt(cond_var[i-1]))
        tomorrow_predicted_var = omega * lt_variance + alpha * (tomorrow_return ** 2) + beta * cond_var[i-1]
        cond_return[i], cond_var[i] = tomorrow_return, tomorrow_predicted_var

    for i in range(1, n_days): ### volatility stochastic process 로 도출 (사실상 garch 에서 return process 를 stochastic 으로 항 정리만 한 셈)

        tomorrow_predicted_var_2 = cond_var_2[i-1] + omega * (lt_variance - cond_var_2[i-1]) + alpha * np.sqrt(2) * cond_var_2[i-1] * np.random.normal(0, 1)
        cond_var_2[i] = tomorrow_predicted_var_2

    cond_vol = np.sqrt(252 * cond_var)

    cond_2_vol = np.sqrt(252 * cond_var_2)

    return cond_return, cond_vol, cond_2_vol


def discrete_kelly(p_vector, outcome_vector):
    
    p_vector = np.array(p_vector)
    
    outcome_vector = np.array(outcome_vector)
    
    # sum of log expected outcomes
    
    def inverted_log_return(kelly_ratio):
        
        result = - np.sum(p_vector * np.log(1 + kelly_ratio * outcome_vector))
        
        return result

    ratio = sciop.minimize(inverted_log_return, x0 = 0.5).x
    expected_return = -sciop.minimize(inverted_log_return, x0 = 0.5).fun
    
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