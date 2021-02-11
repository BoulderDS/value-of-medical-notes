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
from sklearn.externals import joblib

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
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/physician_notes/new_experiments/in_hospital_mortality_24/')
    parser.add_argument('--save_model', type=str, help = 'Path to the directory to store all the trained models',
                        default='/data/physician_notes/new_results/')
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

    union_list_all = []
    union_list_all_but_note = []
    union_list_note = []

    print ("adding BOW features to all and note classifiers .....")

    union_list_all.append(("tfidf", BOWFeatures()))
    union_list_note.append(("tfidf", BOWFeatures()))

    print ("adding Structured features to all and all_but_note classifiers ......")

    union_list_all.append(("structured",
                           Pipeline([
                               ("fe", DictFeatures(structured_features)),
                               ("imputer", SimpleImputer()),
                               ("scaler", MinMaxScaler()),
                           ])))
    union_list_all_but_note.append(("structured",
                           Pipeline([
                               ("fe", DictFeatures(structured_features)),
                               ("imputer", SimpleImputer()),
                               ("scaler", MinMaxScaler()),
                           ])))

    pipeline_unbalanced_all = Pipeline([
        ('union', FeatureUnion(union_list_all)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500)),
    ])

    pipeline_unbalanced_all_but_note = Pipeline([
        ('union', FeatureUnion(union_list_all_but_note)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500)),
    ])

    pipeline_unbalanced_note = Pipeline([
        ('union', FeatureUnion(union_list_note)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500)),
    ])

    pipeline_balanced_all = Pipeline([
        ('union', FeatureUnion(union_list_all)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500,
                                  class_weight="balanced")),
    ])

    pipeline_balanced_all_but_note = Pipeline([
        ('union', FeatureUnion(union_list_all_but_note)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500,
                                  class_weight="balanced")),
    ])

    pipeline_balanced_note = Pipeline([
        ('union', FeatureUnion(union_list_note)),
        ('lr', LogisticRegression(solver="liblinear", max_iter = 500,
                                  class_weight="balanced")),
    ])

    parameters = {
        "lr__C": np.logspace(-5, 5, 11, base = 2)
    }

    t0 = time()
    print ("Training logistic model for unbalanced all features")
    
    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline_unbalanced_all.steps])
    print("parameters:")
    pprint(parameters)
    
    logistic_model_unbalanced_all, logistic_model_unbalanced_all_best_score, logistic_model_unbalanced_all_best_parameters, logistic_model_unbalanced_all_params, logistic_model_unbalanced_all_scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline_unbalanced_all, parameters)
    print ("done in %0.3fs" % (time() - t0))
    
    t0 = time()
    print ("Training logistic model for unbalanced structured features")

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline_unbalanced_all_but_note.steps])
    print("parameters:")
    pprint(parameters)

    logistic_model_unbalanced_all_but_note, logistic_model_unbalanced_all_but_note_best_score, logistic_model_unbalanced_all_but_note_best_parameters, logistic_model_unbalanced_all_but_note_params, logistic_model_unbalanced_all_but_note_scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline_unbalanced_all_but_note, parameters)
    print ("done in %0.3fs" % (time() - t0))

    t0 = time()
    print ("Training logistic model for unbalanced note features")

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline_unbalanced_note.steps])
    print("parameters:")
    pprint(parameters)

    logistic_model_unbalanced_note, logistic_model_unbalanced_note_best_score, logistic_model_unbalanced_note_best_parameters, logistic_model_unbalanced_note_params, logistic_model_unbalanced_note_scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline_unbalanced_note, parameters)
    print ("done in %0.3fs" % (time() - t0))

    t0 = time()
    print ("Training logistic model for balanced all features")

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline_balanced_all.steps])
    print("parameters:")
    pprint(parameters)

    logistic_model_balanced_all, logistic_model_balanced_all_best_score, logistic_model_balanced_all_best_parameters, logistic_model_balanced_all_params, logistic_model_balanced_all_scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline_balanced_all, parameters)
    print ("done in %0.3fs" % (time() - t0))

    t0 = time()
    print ("Training logistic model for balanced structured features")

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline_balanced_all_but_note.steps])
    print("parameters:")
    pprint(parameters)

    logistic_model_balanced_all_but_note, logistic_model_balanced_all_but_note_best_score, logistic_model_balanced_all_but_note_best_parameters, logistic_model_balanced_all_but_note_params, logistic_model_balanced_all_but_note_scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline_balanced_all_but_note, parameters)
    print ("done in %0.3fs" % (time() - t0))

    t0 = time()
    print ("Training logistic model for unbalanced note features")

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline_balanced_note.steps])
    print("parameters:")
    pprint(parameters)

    logistic_model_balanced_note, logistic_model_balanced_note_best_score, logistic_model_balanced_note_best_parameters, logistic_model_balanced_note_params, logistic_model_balanced_note_scores = train_val_compute(train_notes, val_notes, train_y, val_y, pipeline_balanced_note, parameters)
    print ("done in %0.3fs" % (time() - t0))
    print()

    print ("Using All features - unbalanced \n -----------------------------------------------------------------------------------\n")

    print("Best score: %0.3f" % logistic_model_unbalanced_all_best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, logistic_model_unbalanced_all_best_parameters[param_name]))
    print ("Mean test score:")
    print (logistic_model_unbalanced_all_scores)

    test_predicted_unbalanced_all = logistic_model_unbalanced_all.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted_unbalanced_all))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted_unbalanced_all, K))

    print ("Using All But Note features - unbalanced \n -----------------------------------------------------------------------------------\n")

    print("Best score: %0.3f" % logistic_model_unbalanced_all_but_note_best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, logistic_model_unbalanced_all_but_note_best_parameters[param_name]))
    print ("Mean test score:")
    print (logistic_model_unbalanced_all_but_note_scores)

    test_predicted_unbalanced_all_but_note = logistic_model_unbalanced_all_but_note.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted_unbalanced_all_but_note))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted_unbalanced_all_but_note, K))

    print ("Using Note features - unbalanced \n -----------------------------------------------------------------------------------\n")

    print("Best score: %0.3f" % logistic_model_unbalanced_note_best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, logistic_model_unbalanced_note_best_parameters[param_name]))
    print ("Mean test score:")
    print (logistic_model_unbalanced_note_scores)

    test_predicted_unbalanced_note = logistic_model_unbalanced_note.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted_unbalanced_note))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted_unbalanced_note, K))

    print ("Using All features - balanced \n -----------------------------------------------------------------------------------\n")

    print("Best score: %0.3f" % logistic_model_balanced_all_best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, logistic_model_balanced_all_best_parameters[param_name]))
    print ("Mean test score:")
    print (logistic_model_balanced_all_scores)

    test_predicted_balanced_all = logistic_model_balanced_all.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted_balanced_all))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted_balanced_all, K))

    print ("Using All But Note features - balanced \n -----------------------------------------------------------------------------------\n")

    print("Best score: %0.3f" % logistic_model_balanced_all_but_note_best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, logistic_model_balanced_all_but_note_best_parameters[param_name]))
    print ("Mean test score:")
    print (logistic_model_balanced_all_but_note_scores)

    test_predicted_balanced_all_but_note = logistic_model_balanced_all_but_note.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted_balanced_all_but_note))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted_balanced_all_but_note, K))

    print ("Using Note features - balanced \n -----------------------------------------------------------------------------------\n")

    print("Best score: %0.3f" % logistic_model_balanced_note_best_score)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, logistic_model_balanced_note_best_parameters[param_name]))
    print ("Mean test score:")
    print (logistic_model_balanced_note_scores)

    test_predicted_balanced_note = logistic_model_balanced_note.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(test_y, test_predicted_balanced_note))

    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, test_predicted_balanced_note, K))

    print ("Saving all models \n --------------------------------------------------------------------------------------------------\n")

    if args.period_length == 24.0:
        time_period = '24'
    elif args.period_length == 48.0:
        time_period = '48'
    else:
        time_period = 'retro'

    joblib.dump(logistic_model_unbalanced_all, os.path.join(args.save_model, 'unbalanced_all_{}.pkl'.format(time_period)))
    joblib.dump(logistic_model_unbalanced_all_but_note, os.path.join(args.save_model, 'unbalanced_all_but_note_{}.pkl'.format(time_period)))
    joblib.dump(logistic_model_unbalanced_note, os.path.join(args.save_model, 'unbalanced_note_{}.pkl'.format(time_period)))
    joblib.dump(logistic_model_balanced_all, os.path.join(args.save_model, 'balanced_all_{}.pkl'.format(time_period)))
    joblib.dump(logistic_model_balanced_all_but_note, os.path.join(args.save_model, 'balanced_all_but_note_{}.pkl'.format(time_period)))
    joblib.dump(logistic_model_balanced_note, os.path.join(args.save_model, 'balanced_note_{}.pkl'.format(time_period)))
