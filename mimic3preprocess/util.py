#!/usr/bin/env python3

import pandas as pd
import os

def dataframe_from_csv(path, header = 0, index_col = 0):

    if os.path.exists(path + '.gz'):
        return pd.read_csv(path + '.gz', header = header, index_col = index_col, compression = 'gzip')
    return pd.read_csv(path, header = header, index_col = index_col)
    
