#!/usr/bin/env python3

import numpy as np
import os
import random
import pandas as pd

def read_chunk(reader, chunk_size):

    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data['input'] = np.array(data['input'])
    data['masking'] = np.array(data['masking'])
    data['timestamp'] = np.array(data['timestamp'])
    data['label'] = np.array(data['label'])
    return data
