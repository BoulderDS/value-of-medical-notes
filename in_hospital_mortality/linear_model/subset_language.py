#!/usr/bin/env python3

from pprint import pprint
from time import time
import logging

from mimic3preprocess.readers import InHospitalMortalityReader
from in_hospital_mortality.feature_definitions import BOWFeatures, DictFeatures
from in_hospital_mortality.custom_metrics import mortality_rate_at_k, train_val_compute
from in_hospital_mortality import common_utils
from in_hospital_mortality import subset_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

import os
import numpy as np
import pandas as pd
import argparse
import json

def read_and_extract_features(reader):
    structured_data, note_data = common_utils.read_chunk(reader, reader.get_number_of_examples())
    structured_features = common_utils.extract_features_from_rawdata(structured_data['X'], structured_data['header'])
    return (structured_features, note_data, structured_data['y'], structured_data['name'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--period_length', type=float, default=24.0, help='specify the period of prediction',
                        choices=[24.0, 48.0, -1])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/physician_notes/new_experiments/in_hospital_mortality_24/')
    parser.add_argument('--load_model', type=str, help = 'Path to the directory containing all the trained models',
                        default='/data/physician_notes/new_results/')
    args = parser.parse_args()
    args.period_length = float('inf') if args.period_length == -1 else args.period_length
    print (args)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=args.period_length)
    
    if args.period_length == 24.0:
        time_period = '24'
    elif args.period_length == 48.0:
        time_period = '48'
    else:
        time_period = 'retro'

    logistic_model_all = joblib.load(os.path.join(args.load_model, 'balanced_all_{}.pkl'.format(time_period)))
    logistic_model_note = joblib.load(os.path.join(args.load_model, 'balanced_note_{}.pkl'.format(time_period)))

    print('Reading data and extracting features ...')
    (test_X_structured, test_note_data, test_y, test_names) = read_and_extract_features(test_reader)
    print ('Finished reading testing data ...')
    print('  test data shape = {}'.format(test_X_structured.shape))

    structured_features = {}
    for x, name in zip(test_X_structured, test_names):
        structured_features[name] = x

    english, medical = subset_utils.create_english_medical_split(test_note_data['X'])

    english_notes = pd.DataFrame({'file_name': test_names, 'text': english})
    medical_notes = pd.DataFrame({'file_name': test_names, 'text': medical})

    print ("Using the Notes with only extracted English Words\n -----------------------------------------------------------------------------------\n")

    english_predicted_all = logistic_model_all.predict_proba(english_notes)[:, 1]
    english_predicted_note = logistic_model_note.predict_proba(english_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, english_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, english_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, english_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, english_predicted_note, K))

    print ("Using the Notes with only extracted Medical Words\n -----------------------------------------------------------------------------------\n")

    medical_predicted_all = logistic_model_all.predict_proba(medical_notes)[:, 1]
    medical_predicted_note = logistic_model_note.predict_proba(medical_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, medical_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, medical_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, medical_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, medical_predicted_note, K))

    print ("\n\n ------ DONE!!------------------------------------\n")
