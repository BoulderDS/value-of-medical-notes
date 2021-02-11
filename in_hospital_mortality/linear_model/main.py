#!/usr/bin/env python3

from pprint import pprint
from time import time
import logging

from mimic3preprocess.readers import InHospitalMortalityReader
from in_hospital_mortality.feature_definitions import BOWFeatures, DictFeatures
from in_hospital_mortality.custom_metrics import mortality_rate_at_k, train_val_compute
from in_hospital_mortality import common_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

import os
import numpy as np
import pandas as pd
import argparse
import json

def read_and_extract_features(reader):
    structured_data, note_data = common_utils.read_chunk(reader, reader.get_number_of_examples())
    structured_features = common_utils.extract_features_from_rawdata(structured_data['X'], structured_data['header'])
    note_combined = common_utils.combine_note_per_patient(note_data['X'])
    return (structured_features, note_combined, structured_data['y'], structured_data['name'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--period_length', type=float, default=24.0, help='specify the period of prediction',
                        choices=[24.0, 48.0, -1])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'notes', 'all_but_notes'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/physician_notes/new_experiments/in_hospital_mortality_24/')
    parser.add_argument('--balanced', dest="balanced", action="store_true", help = 'whether to use balanced class weights')
    args = parser.parse_args()
    args.period_length = float('inf') if args.period_length == -1 else args.period_length
    print (args)

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=args.period_length)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=args.period_length)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=args.period_length)

    print('Reading data and extracting features ...')
    (train_X_structured, train_X_notes, train_y, train_names) = read_and_extract_features(train_reader)
    print ('Finished reading training data ...')
    (val_X_structured, val_X_notes, val_y, val_names) = read_and_extract_features(val_reader)
    print ('Finished reading validation data ...')
    (test_X_structured, test_X_notes, test_y, test_names) = read_and_extract_features(test_reader)
    print ('Finished reading testing data ...')
    print('  train data shape = {}'.format(train_X_structured.shape))
    print('  validation data shape = {}'.format(val_X_structured.shape))
    print('  test data shape = {}'.format(test_X_structured.shape))

    structured_features = {}
    for x, name in zip(train_X_structured, train_names):
        structured_features[name] = x
    for x, name in zip(val_X_structured, val_names):
        structured_features[name] = x
    for x, name in zip(test_X_structured, test_names):
        structured_features[name] = x
    
    train_notes = pd.DataFrame({'file_name': train_names, 'text': train_X_notes})
    val_notes = pd.DataFrame({'file_name': val_names, 'text': val_X_notes})
    test_notes = pd.DataFrame({'file_name': test_names, 'text': test_X_notes})

    union_list = []
    if args.features in ['all', 'notes']:
        print ("add Bag of Words features .....")
        union_list.append(("tfidf", BOWFeatures()))
    if args.features in ['all','all_but_notes']:
        print ("add structured variable features ..... ")
        union_list.append(("structured",
                           Pipeline([
                               ("fe", DictFeatures(structured_features)),
                               ("imputer", SimpleImputer()),
                               ("scaler", MinMaxScaler()),
                           ])))

    pipeline = Pipeline([
        ('union', FeatureUnion(union_list)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500,
                                  class_weight="balanced" if args.balanced else None)),
    ])

    parameters = {
        "lr__C": np.logspace(-5, 5, 11, base = 2)
    }

    # Display of parameters

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    # Training on training data and hyperparameter tuning on validation data

    t0 = time()
    trained_pipeline, best_score, best_parameters, params, scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline, parameters) 
    print("done in %0.3fs" % (time() - t0))
    print()

    # Displaying training results

    print("Best score: %0.3f" % best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print ("Mean test score:")
    print(scores)

    # Displaying test results

    test_predicted = trained_pipeline.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted, K))
