# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:19:21 2022

@author: Kwan
"""
import pandas as pd
import numpy as np
import scipy.stats as scistat
import scipy.optimize as sciop

    
def call_p(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    N_d1 = scistat.norm.cdf(d1)
    d2 = d1 - v * np.sqrt(t)
    N_d2 = scistat.norm.cdf(d2)
    
    price = s * N_d1 - k * np.exp(-r * t) * N_d2
    
    return price

def put_p(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    N_d1 = scistat.norm.cdf(d1)
    d2 = d1 - v * np.sqrt(t)
    N_d2 = scistat.norm.cdf(d2)
    
    price = k * np.exp(-r * t) * (1 - N_d2) - s * (1 - N_d1)
    
    return price

def call_delta(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    N_d1 = scistat.norm.cdf(d1)
    
    return N_d1

def put_delta(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    N_d1 = scistat.norm.cdf(d1)
    
    return - (1 - N_d1)

def gamma(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    n_d1 = scistat.norm.pdf(d1)
    
    gamma = n_d1 / (s * v * np.sqrt(t))
    
    return gamma

def vega(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    n_d1 = scistat.norm.pdf(d1)
    
    vega = (s * np.sqrt(t) * n_d1)/100
    
    return vega
    
def call_theta(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    
    theta = - (s * v * scistat.norm.pdf(d1) / (2 * np.sqrt(t))) - (r * k * np.exp(-r * t) * scistat.norm.cdf(d2))
    call_theta = theta/365
    
    return call_theta

def put_theta(s, k, v, t, r):
    
    d1 = (np.log(s / k) + (r + np.power(v, 2)/2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    
    theta = - (s * v * scistat.norm.pdf(d1) / (2 * np.sqrt(t))) + (r * k * np.exp(-r * t) * scistat.norm.cdf(-d2))
    put_theta = theta/365
    
    return put_theta
    
    
## 가격 = spot 이 주어졌을때, 개별옵션의 IV 역산하여 도출, minimizing Least Square 로 접근하였음

def derive_iv(s, k, v, t, r, spot, callput = 'call'):
    
    iv = 0
    
    if callput == 'call':
        
        def diff_func_LS(v, s, k, t, r, spot):
            result = np.sum(np.power(call_p(s, k, v, t, r) - spot, 2))
            return result
        
        optimize = sciop.minimize(diff_func_LS, x0 = 0.1, args = (s, k, t, r, spot))
        iv = optimize.x
        
    else: 
        
        def diff_func_LS(v, s, k, t, r, spot):
            result = np.sum(np.power(put_p(s, k, v, t, r) - spot, 2))
            return result
        
        optimize = sciop.minimize(diff_func_LS, x0 = 0.1, args = (s, k, t, r, spot))
        iv = optimize.x
        
    return iv
    

## 헤지스킴 도출 목적으로 주가 각 구간별로 내 현재 포지션의 가격 및 델타 예상치에 대한 df 만들어내는 특수 함수
## 나중에 포지션 projection 툴이랑 엮어서 수정할 예정
    
    
def generate_hedge_df(strike, hedge_interval, s, v, t, r, number, callput = 'call'):
    
    s_array = [s + i * hedge_interval for i in range(-20, 20)]
    v_array = np.repeat(v, len(s_array))
    
    # 주가별로 IV 동일하다는 가정은 사실 비현실적. 주가변화에 따른 개별옵션의 IV도 변화 => IV Curve 변형 없다는 가정 하 IV Curve 가져온 뒤 realized IV 식으로 array 구성
    
    df =  pd.DataFrame(dict(s = s_array, k = strike, v = v_array, t = t, r = r), index = s_array)
    
    if callput == 'call':
        df = df.assign(p = call_p(df.s, df.k, df.v, df.t, df.r), position = call_p(df.s, df.k, df.v, df.t, df.r) * number, delta = call_delta(df.s, df.k, df.v, df.t, df.r), position_delta = call_delta(df.s, df.k, df.v, df.t, df.r) * number)        
    else:
        df = df.assign(p = put_p(df.s, df.k, df.v, df.t, df.r), position = put_p(df.s, df.k, df.v, df.t, df.r) * number, delta = put_delta(df.s, df.k, df.v, df.t, df.r), position_delta = put_delta(df.s, df.k, df.v, df.t, df.r) * number)
 
    return df

    

