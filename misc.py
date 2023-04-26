import scipy.stats as sstat
import numpy as np
import pandas as pd


def moments(df):
    mean = df.mean()
    std = df.std()
    skewness = sstat.skew(df)
    kurtosis = sstat.kurtosis(df)

    result_dict = dict(mean = mean, std = std, skewness = skewness, kurtosis = kurtosis)

    return result_dict

