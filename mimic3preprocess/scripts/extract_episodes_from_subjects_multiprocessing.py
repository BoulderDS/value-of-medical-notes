#!/usr/bin/env python3

import multiprocessing
import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings

from mimic3preprocess.subjects import *
from mimic3preprocess.preprocessing import *

def subprocess(listdir, args):
    items_list = np.load(args.items_list_file)
    print(len(listdir))
    for subject_dir in listdir:

        #directory_name = os.path.join(args.subjects_root_path, subject_dir)
        directory_name = subject_dir
        try:
            print(subject_dir)
            subject_id = int(subject_dir.split("/")[-1])
            print(subject_id)
            if not os.path.isdir(directory_name):
                raise Exception
        except:
            continue
        print ('Subject {}: '.format(subject_id))

        try:
            print ('reading ... ')
            admissions = read_admissions(directory_name)
            events = read_events(directory_name)
        except:
            print ('Error reading from disc!\n')
            continue
        else:
            print ('Finished reading!')


        episodic_data = assemble_episodic_data(admissions)

        #print ('cleaning and converting to time series .... ')

        events = keep_only_items(events, items_list)
        if events.shape[0] == 0:
            print ('No valid events!!\n')
            continue

        timeseries = convert_events_to_timeseries(events, items_list = items_list)

        #print ('extracting seperate episodes ..... ')

        for i in range(admissions.shape[0]):
            hadm_id = admissions['HADM_ID'].iloc[i]
            #print (' {}'.format(hadm_id))
            admittime = admissions['ADMITTIME'].iloc[i]
            dischtime = admissions['DISCHTIME'].iloc[i]
            episode = get_valid_events_for_stay(timeseries, hadm_id, admittime, dischtime)

            if episode.shape[0] == 0:
                print ('No Data!!!!!')
                continue

            episode = add_hours_elpased_to_events(episode, admittime)\
                           .set_index('HOURS')\
                           .sort_index(axis=0)

            episodic_data[episodic_data.index == hadm_id].to_csv(os.path.join(directory_name,'episode{}_outcomes.csv'.format(i+1)), index_label = 'Admission')
            #columns = list(episode.columns)
            #columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
            episode.to_csv(os.path.join(directory_name, 'episode{}_timeseries.csv'.format(i+1)), index_label='Hours')

        print ('DONE!!!!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
    parser.add_argument('subjects_root_path', type=str,
                        default='/data/joe/physician_notes/mimic-data/preprocessed/',
                        help='Directory containing subject sub-directories.')
    parser.add_argument('num_worker', type=int,
                        default=40,
                        help='Number of workers')
    parser.add_argument('--items_list_file', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../resources/itemid_list.npy'),
                        help='Numpy array containing list of ITEMID to consider as features')
    parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    train_dirs = os.listdir(args.subjects_root_path+'train/')
    train_dirs = [os.path.join(args.subjects_root_path,'train',d) for d in train_dirs]
    test_dirs = os.listdir(args.subjects_root_path+'test/')
    test_dirs = [os.path.join(args.subjects_root_path,'test',d) for d in test_dirs]
    all_dirs = train_dirs+test_dirs
    print(len(all_dirs))
    num_dirs = len(all_dirs)
    chunk_size = num_dirs//args.num_worker
    for i in range(args.num_worker+1):
        ## right here
        p = multiprocessing.Process(target=subprocess, args=(all_dirs[chunk_size*i:chunk_size*(i+1)], args))
        p.start()
        print(f"enter worker: {i}")
