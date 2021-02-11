#!/usr/bin/env python3

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mimic3preprocess.mimic3tables import *

def get_train_test_labels(admissions):

    admissions.sort_values(by = ['ADMITTIME'], inplace = True)
    adm = admissions.groupby(['SUBJECT_ID'])['HADM_ID','ADMITTIME','HOSPITAL_EXPIRE_FLAG'].last().reset_index()
    X, y = adm.drop(columns = ['HOSPITAL_EXPIRE_FLAG']), adm['HOSPITAL_EXPIRE_FLAG']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 17, stratify = y)
    X_train['ISTEST'] = 0
    X_test['ISTEST'] = 1
    adm_new = pd.concat([X_train, X_test], ignore_index=True, sort=False)
    adm_new = adm_new[['SUBJECT_ID','ISTEST']]

    return adm_new

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility program to show how train-test split labels were created')
    parser.add_argument('mimic3_path', type = str, help = 'Directory containing the MIMIC-III csv files.')
    parser.add_argument('output_path', type = str, help = 'Directory in which the testset labels should be written.')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    patients = read_patients_table(args.mimic3_path)
    admissions = read_admissions_table(args.mimic3_path)
    icustays = read_icustays_times_in_icu_and_db(args.mimic3_path)

    admissions = merge_on_subject(admissions, patients)
    admissions = merge_on_subject_admits(admissions, icustays)

    admissions = add_age_to_admits(admissions)
    admissions = filter_admits_on_age(admissions)

    admissions = get_train_test_labels(admissions)
    admissions.to_csv(os.path.join(args.output_path, 'testset.csv'), index=False, header=False)
