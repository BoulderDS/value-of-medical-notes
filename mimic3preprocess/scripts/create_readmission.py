#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import re
import random
random.seed(12796)
from mimic3preprocess.subjects import read_admissions

def process_partition(args, partition):
    """
    in this code, we want to generate 30 days readmission data
    """
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))

    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        patient_ts_files = sorted(patient_ts_files, key=lambda f: int(re.sub('\D', '', f)))
        if len(patient_ts_files) <= 1:
            readmission = 0
            try:
                output_ts_filename = patient + "_" + patient_ts_files[0] #only need first ts file
            except:
                print(patient)
                continue
            xy_pairs.append((output_ts_filename, readmission))
        else:
            admit = read_admissions(patient_folder)
            for i in range(len(patient_ts_files)):
                if admit['MORTALITY'].iloc[i] == 1:
                    break
                if i == len(patient_ts_files)-1:
                    readmission = 0
                else:
                    interval = (admit.iloc[i+1]['ADMITTIME']-admit.iloc[i]['DISCHTIME'])/ np.timedelta64(1, 's') /86400 #day
                    if interval <= 30:
                        readmission = 1
                    else:
                        readmission = 0
                try:
                    output_ts_filename = patient + "_" + patient_ts_files[i] #only need first ts file
                except:
                    print(patient)
                    continue
                xy_pairs.append((output_ts_filename, readmission))

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')


    print("\n", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, f"{partition}_retro_listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Create data for readmission prediction task.')
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "train")
    process_partition(args, "test")
