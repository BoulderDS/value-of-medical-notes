#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import gzip

from mimic3preprocess.util import *
from mimic3preprocess.resources import admissions_pb2, events_pb2

def create_admission_df(protobuf_file):

    new_df = []
    for admission in protobuf_file.admissions:
        new_df.append({'SUBJECT_ID': admission.subject_id, 'HADM_ID': admission.hadm_id, 'TIMES_IN_ICU': admission.times_in_icu, 'AGE': admission.age, 'DATE_OF_BIRTH': admission.date_of_birth, 'ADMITTIME': admission.admittime, 'DISCHTIME': admission.dischtime, 'ADMISSION_TYPE': admission.admission_type, 'DIAGNOSIS': admission.diagnosis, 'DBSOURCE': admission.dbsource, 'MORTALITY': admission.hospital_expire_flag})
    new_df = pd.DataFrame(new_df)
    return new_df

def read_admissions(subject_path):

    file_name = os.path.join(subject_path, 'admissions.pb')
    # Read from file
    f = open(file_name, "rb")
    pb = admissions_pb2.Subject()
    pb.ParseFromString(f.read())
    f.close()
    # Convert from pb to dataframe
    admissions = create_admission_df(pb)
    # Convert datatypes
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
    admissions['LOS'] = admissions['DISCHTIME'] - admissions['ADMITTIME']
    admissions['LOS'] = admissions['LOS'].apply(lambda x: x.total_seconds() / 3600)
    admissions.sort_values(by = ['ADMITTIME', 'DISCHTIME'], inplace = True)
    return admissions

def create_events_df(protobuf_file):

    new_df = []
    for event in protobuf_file.events:
        new_df.append({'SUBJECT_ID': event.subject_id, 'HADM_ID': event.hadm_id, 'CHARTTIME': event.charttime, 'CGID': event.cgid, 'ITEMID': event.itemid, 'VALUE': event.value, 'VALUEUOM': event.valueuom})
    new_df = pd.DataFrame(new_df)

    new_df['CHARTTIME'] = pd.to_datetime(new_df['CHARTTIME'])
    return new_df

def read_events(subject_path):

    file_name = os.path.join(subject_path, 'events.csv')
    events = pd.read_csv(file_name)
    events['VALUEUOM'] = events['VALUEUOM'].replace('', np.nan)
    events['VALUE'] = events['VALUE'].replace('', np.nan)
    events = events[~events.VALUE.isna()]
    return events
