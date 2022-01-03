#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np

import random
random.seed(12796)

def process_partition(args, partition, eps = 1e-6):

    n_hours = float('inf') if args.partition_type == -1 else args.partition_type
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))

        for ts_filename in patient_ts_files:
            lb_filename = ts_filename.replace("_timeseries", "_outcomes")
            label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
            # empty label file
            if label_df.shape[0] == 0:
                continue
            mortality = int(label_df.iloc[0]["Mortality"])
            los = label_df.iloc[0]['Length_of_stay']
            if pd.isnull(los):
                print("\n\t(length of stay is missing)", patient, ts_filename)
                continue

            if los < n_hours - eps and n_hours != float('inf'):
                continue

            ts_df = pd.read_csv(os.path.join(patient_folder, ts_filename), low_memory=False)
            #ts_df = ts_df[(ts_df['Hours'] > 1e-6) & (ts_df['Hours'] < n_hours + 1e-6)].reset_index(drop=True)
            if 'CGID' not in ts_df.columns:
                print("\n\t(no CGID) ", patient, ts_filename)
                continue
            ts_df = ts_df[['Hours','900016']].dropna(subset=['900016']).reset_index(drop=True) #physician note
            # no measurements
            if ts_df.shape[0] == 0:
                print("\n\t(no events in Admission) ", patient, ts_filename)
                continue

            output_ts_filename = patient + "_" + ts_filename

            # ts_df.to_csv(os.path.join(output_dir, output_ts_filename), index = False)

            xy_pairs.append((output_ts_filename, mortality))

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')


    print("\n", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Create data for in-hospital mortality prediction task.')
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('partition_type', type=float, help="Enter the timeseries you are considering -> Enter 24 for 24 hours, 48 for 48 hours and -1 for retrospective")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "train")
    process_partition(args, "test")
