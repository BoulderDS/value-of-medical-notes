from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from scipy.stats import skew

def get_admission_range(begin, end, period):
    # first p %
    if period == -1:
        return (begin, end)
    else:
        return (begin, begin+period)


def calculate(channel_data, period, functions):
    if len(channel_data) == 0:
        return np.full((len(functions, )), np.nan)

    L = channel_data[0][0]
    R = channel_data[-1][0]
    L, R = get_admission_range(L, R, period)

    data = [x for (t, x) in channel_data
            if L - 1e-6 < t < R + 1e-6]

    if len(data) == 0:
        return np.full((len(functions, )), np.nan)
    return np.array(data, dtype=np.float32)


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

