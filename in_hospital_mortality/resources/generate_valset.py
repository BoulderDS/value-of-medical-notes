#!/usr/bin/env python3

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def make_train_val_split(args):

    patients = list(filter(str.isdigit, os.listdir(args.root_path)))
    subject_list = []
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, patient)
        patient_label_files = list(filter(lambda x: x.find("outcomes") != -1, os.listdir(patient_folder)))
        hospital_expire_flag = 0

        for label_filename in patient_label_files:
            label_df = pd.read_csv(os.path.join(patient_folder, label_filename))
            if label_df.shape[0] == 0:
                continue
            hospital_expire_flag = hospital_expire_flag | int(label_df.iloc[0]["Mortality"])

        subject_list.append([patient,hospital_expire_flag])

    subjects_df = pd.DataFrame(subject_list, columns = ['SUBJECT_ID', 'MORTALITY'])

    X, y = subjects_df.drop(columns = ['MORTALITY']), subjects_df['MORTALITY']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, random_state = 17, stratify = y)
    X_train['ISVAL'] = 0
    X_val['ISVAL'] = 1
    subject_new = pd.concat([X_train, X_val], ignore_index = True, sort = False)
    subject_new = subject_new[['SUBJECT_ID','ISVAL']]
    subject_new.to_csv(os.path.join(args.output_path, 'valset.csv'), index=False, header=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility program to show how train-test split labels were created')
    parser.add_argument('root_path', type = str, help = 'Directory containing the training data files')
    parser.add_argument('output_path', type = str, help = 'Directory in which the validation set labels should be written.')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    make_train_val_split(args)
