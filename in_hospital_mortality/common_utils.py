#!/usr/bin/env python3

import numpy as np
import os
import json
import random
import pandas as pd

from in_hospital_mortality import feature_extractor

def convert_to_dict(data, header, channel_info):
    '''
    convert data from readers output in to array of arrays format
    '''
    ret = [[] for i in range(data.shape[1] - 1)]
    # The first column is hours
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != '']
        channel = header[i]
        try:
            if len(channel_info[channel]['possible_values']) != 0:
                ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        except KeyError:
            ret[i-1] = list(map(lambda x: (x[0], 0), ret[i-1]))
        try:
            ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
        except ValueError:
            ret[i-1] = list(map(lambda x: (float(x[0]), float(0)), ret[i-1]))
    return ret

def extract_features_from_rawdata(chunk, header):
    with open(os.path.join(os.path.dirname(__file__), "resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return feature_extractor.extract_features(data)

def read_chunk(reader, chunk_size):
    print("read chunk")
    structured_data = {}
    note_data = {}
    for i in range(chunk_size):
        ret_s, ret_n = reader.read_next()
        for k, v in ret_s.items():
            if k not in structured_data:
                structured_data[k] = []
            structured_data[k].append(v)
        for k, v in ret_n.items():
            if k not in note_data:
                note_data[k] = []
            note_data[k].append(v)
    structured_data["header"] = structured_data["header"][0]
    note_data["header"] = note_data["header"][0]
    return structured_data, note_data

def combine_note_per_patient(chunk):

    note_data = [' '.join(' '.join(x for x in row if x != '') for row in note[:,1:]) for note in chunk]
    return note_data

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
