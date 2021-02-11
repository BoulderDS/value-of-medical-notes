from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.stats import skew

all_functions = [min, max, np.mean, np.std, skew, len]

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
               (3, 10), (3, 25), (3, 50)]

def get_admission_range(begin, end, period):
    # first p %
    if period == -1:
        return (begin, end)
    else:
        return (begin, begin+period)

def get_range(begin, end, period):
    # first p %
    if period[0] == 2:
        return (begin, begin + (end - begin) * period[1] / 100.0)
    # last p %
    if period[0] == 3:
        return (end - (end - begin) * period[1] / 100.0, end)

    if period[0] == 0:
        L = begin + period[1]
    else:
        L = end + period[1]

    if period[2] == 0:
        R = begin + period[3]
    else:
        R = end + period[3]

    return (L, R)


def calculate(channel_data, period, sub_period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_admission_range(L, R, period)
    L, R = get_range(L, R, sub_period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array([fn(data) for fn in functions], dtype=np.float32)


def extract_features_single_episode(data_raw, period):
    """
    data_raw : shape = [num_itemID, tuple(hour, value)]
    return : shape = [num_itemID*7*6]
    """
    global sub_periods, all_functions
    extracted_features = [np.concatenate([calculate(data_raw[i], period, sub_period, all_functions)
                                          for sub_period in sub_periods],
                                         axis=0)
                          for i in range(len(data_raw))]
    return np.concatenate(extracted_features, axis=0)

