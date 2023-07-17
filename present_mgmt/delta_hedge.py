# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 01:08:16 2023

@author: kanld
"""

import basic_calc 
import pandas as pd
import numpy as np
import logging


## default 로 돌릴시 변수 재설정

hedge_interval = 1

atm = 317.5  # 처음 등가

s = 317.5  # 시초가
t = 7/365 # 잔존만기=
r = 0.037

# type-in 설정변수

call_short_k = 327.5
call_ext_k = 332.5
call_hedge_k = 322.5

put_short_k = 305
put_ext_k = 300
put_hedge_k = 310

call_short_n = -40
call_ext_n = 40
put_short_n = -40
put_ext_n = 40

# scraped variables

call_short_v = 0.1485
call_ext_v = 0.1485
call_hedge_v = 0.1588

put_short_v = 0.1880
put_ext_v = 0.1989
put_hedge_v = 0.1810

# def generate_hedge_df(strike, hedge_interval, s, v, t, r, number, callput = 'call')
    
## 이거 위에 나중에 UI 입힐때는 통째로 input 함수 활용해서 dictionary {'call_short_k' : 'call_short_number', .... } 구조로 재편해야겠음.   

call_short = basic_calc.generate_hedge_df(call_short_k, hedge_interval, s, call_short_v, t, r, call_short_n, callput = 'call')
call_ext = basic_calc.generate_hedge_df(call_ext_k, hedge_interval, s, call_ext_v, t, r, call_ext_n, callput = 'call')

put_short = basic_calc.generate_hedge_df(put_short_k, hedge_interval, s, put_short_v, t, r, put_short_n, callput = 'put')
put_ext = basic_calc.generate_hedge_df(put_ext_k, hedge_interval, s, put_ext_v, t, r, put_ext_n, callput = 'put')

call_hedge = basic_calc.generate_hedge_df(call_hedge_k, hedge_interval, s, call_hedge_v, t, r, 1, callput = "call")
put_hedge = basic_calc.generate_hedge_df(put_hedge_k, hedge_interval, s, put_hedge_v, t, r, 1, callput = 'put')


delta_sum = call_short['position_delta'] + call_ext['position_delta'] + put_short['position_delta'] + put_ext['position_delta']

call_hedge_delta = call_hedge['delta']
put_hedge_delta = put_hedge['delta']




''' 시나리오가 3개가 있음

    ## '특정 방향으로 물려서' = 위든 아래든 포지션 등가(=ATM) 에서 벗어난 상황을 말함

    1) 특정 방향으로 물려서 시작한 뒤로 계속 해당 방향으로 가는 경우 = 그대로 매수하면 됨
    2) 특정 방향으로 물려서 시작한 뒤로 반등/락해서 주가가 (시초가 ~ 포지션 등가) 사이에서 형성되는 경우 = 아무것도 안 해도 됨
    3) 특정 방향으로 물려서 시작했지만 급반등/락 해서 주가가 포지션 등가를 넘어서 반대방향으로 간 경우 = 포지션 등가 이후부터는 반대방향으로 헤지 실행
'''

if s >= atm:
    open_side_delta = delta_sum[delta_sum.index >= atm] - delta_sum.loc[s] # 1) 물려서 시작한 뒤로 계속 오르는 경우의 상방 헤지델타 계산
    open_side_delta[open_side_delta.index <= s] = 0  # 2) 시초가에서 포지션 등가사이에서는 아무것도 안해도 됨 = 헤지수량 0으로 설정
    opposite_side_delta = delta_sum[delta_sum.index < atm] # 3) 반락해서 포지션 등가 반대로 떨어지는 경우의 델타
    
    hedge_with_call = open_side_delta / call_hedge_delta # 아예 이단계에서 옵션으로 헤지하는 상황 구하기
    hedge_with_call.where(hedge_with_call < 0, 0, inplace = True)
    hedge_with_put = opposite_side_delta / put_hedge_delta   
    hedge_with_put.where(hedge_with_put < 0, 0, inplace = True)
else:
    open_side_delta = delta_sum[delta_sum.index <= atm] - delta_sum.loc[s] 
    open_side_delta[open_side_delta.index >= s ] = 0
    opposite_side_delta = delta_sum[delta_sum.index > atm]
    
    hedge_with_put = open_side_delta / put_hedge_delta
    hedge_with_put.where(hedge_with_put < 0, 0, inplace = True)
    hedge_with_call = opposite_side_delta / call_hedge_delta
    hedge_with_call.where(hedge_with_call < 0, 0, inplace = True)
# 선물로 하는 경우
    
delta_sum_future= pd.concat([open_side_delta, opposite_side_delta], axis = 0)

delta_sum_future.sort_index(ascending = True, inplace = True)

hedge_with_future = ( - delta_sum_future).apply(round)  # 선물로 하는 경우 : 그냥 산출된 델타 * -1 + round 하면 헤지해야하는 수량 나옴

# 특정 헤지옵션으로 하는 경우

delta_sum_option = pd.concat([hedge_with_call, hedge_with_put], axis = 1, join = 'inner').fillna(0)

hedge_with_option = - delta_sum_option.apply(round)

hedge_with_option.columns = ['call_hedge', 'put_hedge']

print(hedge_with_future, hedge_with_option, sep = '\n\n\n')

delta_concat = pd.concat([call_ext['position_delta'], call_short['position_delta'], put_short['position_delta'], put_ext['position_delta'], call_hedge_delta, put_hedge_delta, delta_sum, hedge_with_option], axis = 1)