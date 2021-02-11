#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import re
import csv
import gzip
import sys
import warnings
from datetime import datetime
import spacy
from spacy.lang.en import English
from tqdm import tqdm
from multiprocessing import  Pool
from mimic3preprocess.util import *
from mimic3preprocess.resources import events_pb2, admissions_pb2

def read_patients_table(mimic3_path):

    patients = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    patients = patients[['SUBJECT_ID','GENDER','DOB']]
    patients['DOB'] = pd.to_datetime(patients['DOB'])
    return patients

def is_organ_donor(row):

    return (row['ADMITTIME'] > row['DISCHTIME'])

def read_admissions_table(mimic3_path):

    admissions = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
    admissions['IS_ORGAN_DONOR'] = admissions.apply(is_organ_donor, axis = 1)
    admissions['IS_ORGAN_DONOR'] = admissions['IS_ORGAN_DONOR'].map({True:1, False:0})
    admissions = admissions[admissions['HAS_CHARTEVENTS_DATA'] == 1]
    admissions = admissions[admissions['IS_ORGAN_DONOR'] == 0]
    admissions = admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','ADMISSION_TYPE','DIAGNOSIS','HOSPITAL_EXPIRE_FLAG']]
    return admissions

def read_icustays_times_in_icu_and_db(mimic3_path):

    icustays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    icuinfo = icustays.copy()
    icuinfo = icuinfo.groupby(['SUBJECT_ID','HADM_ID'])['ICUSTAY_ID'].count().to_frame().reset_index()
    icuinfo = icuinfo.rename(columns = {'ICUSTAY_ID':'TIMES_IN_ICU'})
    icustays = icustays[['SUBJECT_ID','HADM_ID','DBSOURCE']]
    icustays = merge_on_subject_admits(icustays, icuinfo)
    return icustays

def merge_on_subject(table1, table2):

    return pd.merge(table1, table2, on = ['SUBJECT_ID'])

def merge_on_subject_admits(table1, table2):

    return pd.merge(table1, table2, on = ['SUBJECT_ID', 'HADM_ID'])

def add_age_to_admits(admits):
    admits['ADMITTIME'] = pd.to_datetime(admits['ADMITTIME'], errors='coerce').dt.date
    admits['DOB'] = pd.to_datetime(admits['DOB'], errors='coerce').dt.date
    admits['AGE'] = (admits['ADMITTIME'] - admits['DOB']).astype('timedelta64[D]')/ np.timedelta64(365, 'D')
    admits.loc[admits['AGE'] < 0, 'AGE'] = 90
    return admits

def filter_admits_on_age(admits, min_age = 18, max_age = np.inf):

    admits = admits.loc[(admits['AGE'] >= min_age) & (admits['AGE'] <= max_age)]
    return admits

def get_admits_protobuf(admits):

    admits['ADMITTIME'] = admits['ADMITTIME'].astype('str')
    admits['DISCHTIME'] = admits['DISCHTIME'].astype('str')
    admits['DOB'] = admits['DOB'].astype('str')
    pb = admissions_pb2.Subject()
    for index, row in admits.iterrows():
        adm = pb.admissions.add()
        adm.subject_id = row['SUBJECT_ID']
        adm.hadm_id = row['HADM_ID']
        adm.times_in_icu = row['TIMES_IN_ICU']
        adm.age = row['AGE']
        adm.date_of_birth = row['DOB']
        adm.admittime = row['ADMITTIME']
        adm.dischtime = row['DISCHTIME']
        adm.admission_type = row['ADMISSION_TYPE']
        if row['DIAGNOSIS'] not in row:
            adm.diagnosis = 'NA'
        else:
            adm.diagnosis = row['DIAGNOSIS']
        adm.dbsource = row['DBSOURCE']
        adm.hospital_expire_flag = row['HOSPITAL_EXPIRE_FLAG']
    return pb

def get_events_protobuf_from_ordered_dict(current_observations):

    pb = events_pb2.Episode()
    for row in current_observations:
        event = pb.events.add()
        event.subject_id = int(row['SUBJECT_ID'])
        event.hadm_id = int(row['HADM_ID'])
        event.charttime = row['CHARTTIME']
        if not row['CGID']:
            event.cgid = np.nan
        else:
            event.cgid = float(row['CGID'])
        event.itemid = int(row['ITEMID'])
        event.value = row['VALUE']
        event.valueuom = row['VALUEUOM']
    return pb

def get_events_protobuf_from_df(df):

    pb = events_pb2.Episode()
    for index, row in df.iterrows():
        event = pb.events.add()
        event.subject_id = int(row['SUBJECT_ID'])
        event.hadm_id = int(row['HADM_ID'])
        event.charttime = row['CHARTTIME']
        event.cgid = row['CGID']
        event.itemid = int(row['ITEMID'])
        if pd.isnull(row['VALUE']):
            event.value = ''
        else:
            event.value = str(row['VALUE'])
        if pd.isnull(row['VALUEUOM']):
            event.valueuom = ''
        else:
            event.valueuom = str(row['VALUEUOM'])
    return pb

def break_up_admissions_by_subject(admissions, output_path, subjects = None, verbose = 1):

    admissions.drop_duplicates(inplace=True)
    subjects = admissions['SUBJECT_ID'].unique() if subjects is None else subjects
    num_subjects = subjects.shape[0]
    for i, subject_id in enumerate(subjects):
        if verbose:
            print ('\rSUBJECT {} of {} ....'.format(i+1, num_subjects))
        directory_name = os.path.join(output_path, str(subject_id))

        try:
            os.makedirs(directory_name)
        except:
            pass

        admit_subset = admissions[admissions['SUBJECT_ID'] == subject_id].sort_values(by = 'ADMITTIME')
        admit_pb = get_admits_protobuf(admit_subset)
        f = open(os.path.join(directory_name, 'admissions.pb'), "wb")
        f.write(admit_pb.SerializeToString())
        f.close()
    if verbose:
        print ('DONE!\n')

def break_up_and_write_events(events_df, output_path, subjects, verbose):

    num_subjects = subjects.shape[0]
    events_df.drop_duplicates(inplace=True)
    for i, subject_id in enumerate(subjects):
        if verbose:
            print ('\rProcessing SUBJECT_ID {} -> {} of total {} subjects .....'.format(subject_id, i+1, num_subjects))
        directory_name = os.path.join(output_path, str(subject_id))

        try:
            os.makedirs(directory_name)
        except:
            pass

        event_subset = events_df[events_df['SUBJECT_ID'] == subject_id]
        event_pb = get_events_protobuf_from_df(event_subset)
        file_name = os.path.join(directory_name, 'events.pb.gz')
        f = gzip.open(file_name, "ab")
        f.write(event_pb.SerializeToString())
        f.close()

def process_notevents_get_charttime(row):

    if (row['CATEGORY'] == 'ECG'):
        row['CHARTTIME'] = row['CHARTDATE']
    elif (row['CATEGORY'] == 'Echo'):
        row['CHARTTIME'] = str(datetime.strptime(re.search('Date/Time: (.*)\n', row['TEXT']).group(1), '[**%Y-%m-%d**] at %H:%M'))
    elif (pd.isnull(row['CHARTTIME'])):
        row['CHARTTIME'] = row['STORETIME']

    return row['CHARTTIME']

def read_events_table_by_row(mimic3_path, table):

    num_rows = {'CHARTEVENTS': 330712483, 'INPUTEVENTS_MV': 3618991}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv')))

    if table.upper() == 'INPUTEVENTS_MV':
        reader.fieldnames = 'ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTTIME', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM', 'RATE', 'RATEUOM', 'STORETIME', 'CGID', 'ORDERID', 'LINKORDERID', 'ORDERCATEGORYNAME', 'SECONDARYORDERCATEGORYNAME', 'ORDERCOMPONENTTYPEDESCRIPTION', 'ORDERCATEGORYDESCRIPTION', 'PATIENTWEIGHT', 'TOTALAMOUNT', 'TOTALAMOUNTUOM', 'ISOPENBAG', 'CONTINUEINNEXTDEPT', 'CANCELREASON', 'STATUSDESCRIPTION', 'COMMENTS_EDITEDBY', 'COMMENTS_CANCELEDBY', 'COMMENTS_DATE', 'ORIGINALAMOUNT', 'ORIGINALRATE'

    for i, row in enumerate(reader):
        yield row, i, num_rows[table.upper()]

def parallelize_dataframe(df, func, n_cores):
    print (f"\n Parallel in {n_cores} workers >")
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def tokenize_dataframe_verbose(df):
    tqdm.pandas()
    nlp = spacy.load('en_core_web_sm', parser = False, entity = False, tagger = False)
    tokenizer = English().Defaults.create_tokenizer(nlp)
    df['VALUE'] = df['VALUE'].progress_apply(lambda x: ' '.join([token.text for token in tokenizer(x)]))
    return df

def tokenize_dataframe(df):
    nlp = spacy.load('en_core_web_sm', parser = False, entity = False, tagger = False)
    tokenizer = English().Defaults.create_tokenizer(nlp)
    df['VALUE'] = df['VALUE'].apply(lambda x: ' '.join([token.text for token in tokenizer(x)]))
    return df

def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path, subjects_to_keep = None, verbose=1, n_workers=8):

    obs_header = ['SUBJECT_ID', 'HADM_ID', 'CGID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']

    if subjects_to_keep is not None and table.upper() in ['CHARTEVENTS', 'INPUTEVENTS_MV']:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):

        def __init__(self):

            self.curr_subject_id = ''
            self.last_write_num = 0
            self.last_write_num_rows = 0
            self.last_write_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        data_stats.last_write_num += 1
        data_stats.last_write_num_rows = len(data_stats.curr_obs)
        data_stats.last_write_subject_id = data_stats.curr_subject_id
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    if (table.upper() in ['CHARTEVENTS', 'INPUTEVENTS_MV']):

        for row, row_num, num_rows in read_events_table_by_row(mimic3_path, table):

            if verbose and (row_num % 100000 == 0):
                if data_stats.last_write_num != '':
                    print ('\rProcessing {0}: ROW {1} of {2} .... last write '
                           '({3}) {4} rows for subject {5}'.format(table, row_num, num_rows,
                                                                         data_stats.last_write_num,
                                                                         data_stats.last_write_num_rows,
                                                                         data_stats.last_write_subject_id))
                else:
                    print ('\rProcessing {0}: ROW {1} of {2} .... '.format(table, row_num, num_rows))

            if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
                continue

            row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                       'HADM_ID': row['HADM_ID'],
                       'CHARTTIME': row['CHARTTIME'],
                       'CGID': row['CGID'],
                       'ITEMID': row['ITEMID'],
                       'VALUE': row['VALUE'],
                       'VALUEUOM': row['VALUEUOM']}

            if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
                write_current_observations()
            data_stats.curr_obs.append(row_out)
            data_stats.curr_subject_id = row['SUBJECT_ID']

        if data_stats.curr_subject_id != '':
            write_current_observations()

        if verbose:
            print ('\rProcessing {0}: ROW {1} of {2} .... last write'
                   '({3}) {4} rows for subject {5}'.format(table, row_num, num_rows,
                                                           data_stats.last_write_num,
                                                           data_stats.last_write_num_rows,
                                                           data_stats.last_write_subject_id))

    elif (table.upper() in ['LABEVENTS','INPUTEVENTS_CV','OUTPUTEVENTS']):

        warnings.filterwarnings('ignore')
        print ('\rProcessing {} ->'.format(table.upper()))
        events_df = dataframe_from_csv(os.path.join(mimic3_path, table.upper() + '.csv'))
        events_df = events_df[~events_df['HADM_ID'].isna()]
        if (table.upper() == 'LABEVENTS'):
            events_df['CGID'] = np.nan
        elif (table.upper() == 'INPUTEVENTS_CV'):
            events_df.rename(columns = {'AMOUNT':'VALUE', 'AMOUNTUOM':'VALUEUOM'}, inplace = True)
        events_df = events_df[['SUBJECT_ID','HADM_ID','CHARTTIME','CGID','ITEMID','VALUE','VALUEUOM']]
        events_df['CHARTTIME'] = events_df['CHARTTIME'].astype(str)
        for row_num in range(len(events_df)):
            row = events_df.iloc[row_num]
            if verbose and (row_num % 100000 == 0):
                if data_stats.last_write_num != '':
                    print ('\rProcessing {0}: ROW {1} of {2} .... last write '
                           '({3}) {4} rows for subject {5}'.format(table, row_num, len(events_df),
                                                                         data_stats.last_write_num,
                                                                         data_stats.last_write_num_rows,
                                                                         data_stats.last_write_subject_id))
                else:
                    print ('\rProcessing {0}: ROW {1} of {2} .... '.format(table, row_num, len(events_df)))
            if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
                continue

            row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                       'HADM_ID': row['HADM_ID'],
                       'CHARTTIME': row['CHARTTIME'],
                       'CGID': row['CGID'],
                       'ITEMID': row['ITEMID'],
                       'VALUE': row['VALUE'],
                       'VALUEUOM': row['VALUEUOM']}

            if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
                write_current_observations()
            data_stats.curr_obs.append(row_out)
            data_stats.curr_subject_id = row['SUBJECT_ID']

        if data_stats.curr_subject_id != '':
            write_current_observations()

        if verbose:
            print ('\rProcessing {0}: ROW {1} of {2} .... last write'
                   '({3}) {4} rows for subject {5}'.format(table, row_num, len(events_df),
                                                           data_stats.last_write_num,
                                                           data_stats.last_write_num_rows,
                                                           data_stats.last_write_subject_id))

        print ('\r{} Done!!!'.format(table.upper()))

    elif table.upper() == 'NOTEEVENTS':

        warnings.filterwarnings('ignore')

        # nlp = spacy.load('en_core_web_sm', parser = False, entity = False, tagger = False)
        # tokenizer = English().Defaults.create_tokenizer(nlp)

        print ('\rProcessing {} ->'.format(table.upper()))
        noteevents = dataframe_from_csv(os.path.join(mimic3_path, table.upper() +'.csv'))
        noteevents = noteevents[noteevents['ISERROR'].isna()]
        noteevents = noteevents[~noteevents['HADM_ID'].isna()]
        #noteevents = noteevents[noteevents['CATEGORY'] == 'Discharge summary']

        print ("\rCalculating Charttimes for records in which it is missing")
        if verbose:
            tqdm.pandas()
            noteevents['CHARTTIME'] = noteevents.progress_apply(process_notevents_get_charttime, axis = 1)
        else:
            noteevents['CHARTTIME'] = noteevents.apply(process_notevents_get_charttime, axis = 1)
        noteevents['CHARTTIME'] = pd.to_datetime(noteevents['CHARTTIME'])
        noteevents['CHARTTIME'] = noteevents['CHARTTIME'].astype(str)

        noteevents['CATEGORY'] = noteevents['CATEGORY'].map({'Nursing/other': 900001, 'Physician ': 900002, 'Nutrition': 900003, 'General': 900004, 'Nursing': 900005, 'Respiratory ': 900006,'Rehab Services': 900007, 'Social Work': 900008, 'Echo': 900010,'ECG': 900011,'Case Management ': 900012,'Pharmacy': 900013,'Consult': 900014, 'Radiology': 900015, 'Discharge summary': 900016})
        noteevents.rename(columns = {'CATEGORY':'ITEMID','TEXT':'VALUE'}, inplace = True)
        noteevents['VALUEUOM'] = 'note'

        print ("\rTokenizing Notes beforehand->")



        if verbose:
            noteevents = parallelize_dataframe(noteevents, tokenize_dataframe_verbose, n_workers)
        else:
            noteevents = parallelize_dataframe(noteevents, tokenize_dataframe, n_workers)


        for row_num in range(len(noteevents)):
            row = noteevents.iloc[row_num]

            if verbose and (row_num % 100000 == 0):
                if data_stats.last_write_num != '':
                    print ('\rProcessing {0}: ROW {1} of {2} .... last write '
                           '({3}) {4} rows for subject {5}'.format(table, row_num, len(noteevents),
                                                                         data_stats.last_write_num,
                                                                         data_stats.last_write_num_rows,
                                                                         data_stats.last_write_subject_id))
                else:
                    print ('\rProcessing {0}: ROW {1} of {2} .... '.format(table, row_num, len(noteevents)))
            if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
                continue

            row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                       'HADM_ID': row['HADM_ID'],
                       'CHARTTIME': row['CHARTTIME'],
                       'CGID': row['CGID'],
                       'ITEMID': row['ITEMID'],
                       'VALUE': row['VALUE'],
                       'VALUEUOM': row['VALUEUOM']}

            if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
                write_current_observations()
            data_stats.curr_obs.append(row_out)
            data_stats.curr_subject_id = row['SUBJECT_ID']

        if data_stats.curr_subject_id != '':
            write_current_observations()

        if verbose:
            print ('\rProcessing {0}: ROW {1} of {2} .... last write'
                   '({3}) {4} rows for subject {5}'.format(table, row_num, len(noteevents),
                                                           data_stats.last_write_num,
                                                           data_stats.last_write_num_rows,
                                                           data_stats.last_write_subject_id))

        print ('\r{} Done!!!'.format(table.upper()))
