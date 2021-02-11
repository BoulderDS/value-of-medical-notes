#!/usr/bin/env python3

import numpy as np
import pandas as pd

from mimic3preprocess.util import *
from mimic3preprocess.resources import events_pb2, admissions_pb2

def transform_dbsource(dbseries):

    dbsource_map = {'carevue': 1, 'metavision': 2, 'both': 3}
    return { 'DBSOURCE': dbseries.apply(lambda x: dbsource_map[x]) }

def assemble_episodic_data(admissions):

    data = {'Admission': admissions['HADM_ID'], 'Times_in_ICU': admissions['TIMES_IN_ICU'], 'Age': admissions['AGE'], 'Length_of_stay': admissions['LOS'], 'Mortality': admissions['MORTALITY']}
    data.update(transform_dbsource(admissions['DBSOURCE']))
    data = pd.DataFrame(data).set_index('Admission')
    data = data[['Times_in_ICU','Age','Length_of_stay','Mortality']]
    return data

def keep_only_items(events, items_list):

    return events[events['ITEMID'].isin(items_list)]

def convert_events_to_timeseries(events, items_list = []):

    metadata = events[['CHARTTIME','HADM_ID', 'CGID']]\
                      .sort_values(by = ['CHARTTIME','HADM_ID'])\
                      .drop_duplicates(keep = 'first').set_index('CHARTTIME')
    timeseries = events[['CHARTTIME', 'ITEMID', 'VALUE']]\
                       .sort_values(by=['CHARTTIME', 'ITEMID', 'VALUE'], axis=0)\
                       .drop_duplicates(subset=['CHARTTIME', 'ITEMID'], keep='last')
    timeseries = timeseries\
                       .pivot(index='CHARTTIME', columns='ITEMID', values='VALUE')\
                       .merge(metadata, left_index=True,right_index=True)\
                       .sort_index(axis=0).reset_index()

    for item in items_list:
        if item not in timeseries:
            timeseries[item] = np.nan
    if 'CGID' not in timeseries.columns:
        print(timeseries)
        import sys
        sys.exit()
    return timeseries

def get_valid_events_for_stay(events, hadm_id, admittime = None, dischtime = None):

    idx = (events['HADM_ID'] == hadm_id)
    if admittime is not None and dischtime is not None:
        idx = idx | ((pd.to_datetime(events['CHARTTIME']) >= admittime) & (pd.to_datetime(events['CHARTTIME']) <= dischtime))
    events = events[idx]
    del events['HADM_ID']
    return events

def add_hours_elpased_to_events(events, admittime, remove_charttime=True):

    events['HOURS'] = (pd.to_datetime(events['CHARTTIME']) - admittime).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['CHARTTIME']
    return events
