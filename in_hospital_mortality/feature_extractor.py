#!/usr/bin/env python3

import numpy as np
from scipy.stats import skew

'''
What this program does is that it takes timeseries for single episode feature and converts them to different aggregate functions
and also with different sub periods.

So we have 767 features * 6 aggregate functions * 7 sub_periods = 32214 features
'''

functions = [min, max, np.mean, np.std, skew, len]

'''
What this sub_periods array means ->

(1, 100) - Take all of the time series
(1, 10)  - Take the first 10 % of the time series
(1, 25)  - Take the first 25 % of the time series
(1, 50)  - Take the first 50 % of the time series
(2, 10)  - Take the last  10 % of the time series
(2, 25)  - Take the last  25 % of the time series
(2, 50)  - Take The last  50 % of the time series
'''

sub_periods = [(1, 100), (1, 10), (1, 25), (1, 50), (2, 10), (2, 25), (2, 50)]


def get_range(begin, end, period):

    # First % of timeseries elements
    if period[0] == 1:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # Last % of timeseries elements
    if period[0] == 2:
        return (end - (end - begin) * period[1] / 100.0, end)

def calculate(channel_data, sub_period, functions):

    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    # F is the first timeseries element
    F = channel_data[0][0]
    # L is the last timeseries element
    L = channel_data[-1][0]

    F, L = get_range(F, L, sub_period)

    data = [x for (t, x) in channel_data
            if F - 1e-6 < t < L + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)

def extract_features_single_episode(data_raw, functions):
    global sub_periods
    extracted_features = [np.concatenate([calculate(data_raw[i], sub_period, functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)

def extract_features(data_raw):

    global functions
    return np.array([extract_features_single_episode(x, functions)
                     for x in data_raw])
