# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 02:07:44 2023

@author: kanld
"""

import pandas as pd
import scipy.stats as scistat
import numpy as np
import scipy.optimize as sciop


def gbm_stock(spot, drift, vol, n_days, n_paths = 1):
    
        dt = 1/365
        
        growth_path = np.exp((drift - vol **2 /2) * dt + vol * scistat.norm.rvs(0, np.sqrt(dt), size = [n_days, n_paths]))
    
        growth_path = np.cumprod(growth_path, axis = 0)
        
        price_path = spot * growth_path
        
        return price_path
        

def discrete_kelly(p_vector, outcome_vector):
    
    p_vector = np.array(p_vector)
    
    outcome_vector = np.array(outcome_vector)
    
    # sum of logarithmized expected outcomes
    
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