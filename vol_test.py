import numpy as np
import pandas as pd
import scipy.optimize as sciop

df_raw = pd.read_csv("C:/Users/kanld/Desktop/1.csv")
df_raw = df_raw.applymap(float)

param = [0, 0, 0]

# param[0] = omega for long-term variance;
# param[1] = alpha for deterministic term;
# param[2] = beta for stochastic term

def dummy_optimization(param, df_raw, long_term_vol):

    long_term_variance = (long_term_vol ** 2) / 252

    ''' long_term variance is empirical in nature, rather than being computed from averaging variances'''

    df_model_value = param[0] * long_term_variance + param[1] * df_raw['variance'].shift(1) + param[2] * df_raw['to_daily'].shift(1)

    sse = np.power((df_raw['to_daily'] - df_model_value), 2).sum()

    return sse


a = sciop.minimize(dummy_optimization, x0 = [0.0, 0.0, 0.0], args = (df_raw, 0.20), constraints = ({'type':'eq', 'fun': lambda x : sum(x)- 1}))

print(a)

from arch import arch_model