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

    nursing_other, physician, nutrition, general, nursing, respiratory, rehab_services, social_work, echo, ecg, case_management, pharmacy, consult, radiology = subset_utils.create_seperate_note_cat(test_note_data['X'])
    english, non_english = subset_utils.create_english_medical_split(test_note_data['X'])
    noun, proper_noun, adjective, verb = subset_utils.create_pos_tag_split(test_note_data['X'])
    non_copy, copy = subset_utils.remove_copy_pasting_notes(test_note_data['X'])

    # Converting all subset of notes to dataframes

    nursing_other_notes = pd.DataFrame({'file_name': test_names, 'text': nursing_other})
    physician_notes = pd.DataFrame({'file_name': test_names, 'text': physician})
    nutrition_notes = pd.DataFrame({'file_name': test_names, 'text': nutrition})
    general_notes = pd.DataFrame({'file_name': test_names, 'text': general})
    nursing_notes = pd.DataFrame({'file_name': test_names, 'text': nursing})
    respiratory_notes = pd.DataFrame({'file_name': test_names, 'text': respiratory})
    rehab_services_notes = pd.DataFrame({'file_name': test_names, 'text': rehab_services})
    social_work_notes = pd.DataFrame({'file_name': test_names, 'text': social_work})
    echo_notes = pd.DataFrame({'file_name': test_names, 'text': echo})
    ecg_notes = pd.DataFrame({'file_name': test_names, 'text': ecg})
    case_management_notes = pd.DataFrame({'file_name': test_names, 'text': case_management})
    pharmacy_notes = pd.DataFrame({'file_name': test_names, 'text': pharmacy})
    consult_notes = pd.DataFrame({'file_name': test_names, 'text': consult})
    radiology_notes = pd.DataFrame({'file_name': test_names, 'text': radiology})
    
    english_notes = pd.DataFrame({'file_name': test_names, 'text': english})
    non_english_notes = pd.DataFrame({'file_name': test_names, 'text': non_english})
    
    noun_notes = pd.DataFrame({'file_name': test_names, 'text': noun})
    proper_noun_notes = pd.DataFrame({'file_name': test_names, 'text': proper_noun})
    adjective_notes = pd.DataFrame({'file_name': test_names, 'text': adjective})
    verb_notes = pd.DataFrame({'file_name': test_names, 'text': verb})

    non_copy_notes = pd.DataFrame({'file_name': test_names, 'text': non_copy})
    copy_notes = pd.DataFrame({'file_name': test_names, 'text': copy})

    print ("Using Nursing/Other notes \n -----------------------------------------------------------------------------------\n")
    
    nursing_other_predicted_all = logistic_model_all.predict_proba(nursing_other_notes)[:, 1]
    nursing_other_predicted_note = logistic_model_note.predict_proba(nursing_other_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, nursing_other_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, nursing_other_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, nursing_other_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, nursing_other_predicted_note, K))

    print ("Using Physician notes \n -----------------------------------------------------------------------------------\n")

    physician_predicted_all = logistic_model_all.predict_proba(physician_notes)[:, 1]
    physician_predicted_note = logistic_model_note.predict_proba(physician_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, physician_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, physician_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, physician_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, physician_predicted_note, K))

    print ("Using Nutrition notes \n -----------------------------------------------------------------------------------\n")

    nutrition_predicted_all = logistic_model_all.predict_proba(nutrition_notes)[:, 1]
    nutrition_predicted_note = logistic_model_note.predict_proba(nutrition_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, nutrition_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, nutrition_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, nutrition_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, nutrition_predicted_note, K))

    print ("Using General notes \n -----------------------------------------------------------------------------------\n")

    general_predicted_all = logistic_model_all.predict_proba(general_notes)[:, 1]
    general_predicted_note = logistic_model_note.predict_proba(general_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, general_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, general_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, general_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, general_predicted_note, K))

    print ("Using Nursing notes \n -----------------------------------------------------------------------------------\n")

    nursing_predicted_all = logistic_model_all.predict_proba(nursing_notes)[:, 1]
    nursing_predicted_note = logistic_model_note.predict_proba(nursing_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, nursing_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, nursing_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, nursing_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, nursing_predicted_note, K))

    print ("Using Respiratory notes \n -----------------------------------------------------------------------------------\n")

    respiratory_predicted_all = logistic_model_all.predict_proba(respiratory_notes)[:, 1]
    respiratory_predicted_note = logistic_model_note.predict_proba(respiratory_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, respiratory_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, respiratory_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, respiratory_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, respiratory_predicted_note, K))

    print ("Using Rehab Services notes \n -----------------------------------------------------------------------------------\n")

    rehab_services_predicted_all = logistic_model_all.predict_proba(rehab_services_notes)[:, 1]
    rehab_services_predicted_note = logistic_model_note.predict_proba(rehab_services_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, rehab_services_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, rehab_services_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, rehab_services_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, rehab_services_predicted_note, K))

    print ("Using Social Work notes \n -----------------------------------------------------------------------------------\n")

    social_work_predicted_all = logistic_model_all.predict_proba(social_work_notes)[:, 1]
    social_work_predicted_note = logistic_model_note.predict_proba(social_work_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, social_work_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, social_work_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, social_work_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, social_work_predicted_note, K))

    print ("Using Echo notes \n -----------------------------------------------------------------------------------\n")

    echo_predicted_all = logistic_model_all.predict_proba(echo_notes)[:, 1]
    echo_predicted_note = logistic_model_note.predict_proba(echo_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, echo_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, echo_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, echo_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, echo_predicted_note, K))

    print ("Using ECG notes \n -----------------------------------------------------------------------------------\n")

    ecg_predicted_all = logistic_model_all.predict_proba(ecg_notes)[:, 1]
    ecg_predicted_note = logistic_model_note.predict_proba(ecg_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, ecg_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, ecg_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, ecg_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, ecg_predicted_note, K))

    print ("Using Case Management notes \n -----------------------------------------------------------------------------------\n")

    case_management_predicted_all = logistic_model_all.predict_proba(case_management_notes)[:, 1]
    case_management_predicted_note = logistic_model_note.predict_proba(case_management_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, case_management_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, case_management_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, case_management_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, case_management_predicted_note, K))

    print ("Using Pharmacy notes \n -----------------------------------------------------------------------------------\n")

    pharmacy_predicted_all = logistic_model_all.predict_proba(pharmacy_notes)[:, 1]
    pharmacy_predicted_note = logistic_model_note.predict_proba(pharmacy_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, pharmacy_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, pharmacy_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, pharmacy_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, pharmacy_predicted_note, K))

    print ("Using Consult notes \n -----------------------------------------------------------------------------------\n")

    consult_predicted_all = logistic_model_all.predict_proba(consult_notes)[:, 1]
    consult_predicted_note = logistic_model_note.predict_proba(consult_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, consult_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, consult_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, consult_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, consult_predicted_note, K))

    print ("Using Radiology notes \n -----------------------------------------------------------------------------------\n")

    radiology_predicted_all = logistic_model_all.predict_proba(radiology_notes)[:, 1]
    radiology_predicted_note = logistic_model_note.predict_proba(radiology_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, radiology_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, radiology_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, radiology_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, radiology_predicted_note, K))

    print ("\n\n -------------------------------------------------------------------------------------------------------\n")

    print ("Using English notes \n -----------------------------------------------------------------------------------\n")

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

    print ("Using Non-English (Medical) notes \n -----------------------------------------------------------------------------------\n")

    non_english_predicted_all = logistic_model_all.predict_proba(non_english_notes)[:, 1]
    non_english_predicted_note = logistic_model_note.predict_proba(non_english_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, non_english_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, non_english_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, non_english_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, non_english_predicted_note, K))

    print ("\n\n ----------------------------------------------------------------------------------------------------------------\n")

    print ("Using Noun notes \n -----------------------------------------------------------------------------------\n")

    noun_predicted_all = logistic_model_all.predict_proba(noun_notes)[:, 1]
    noun_predicted_note = logistic_model_note.predict_proba(noun_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, noun_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, noun_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, noun_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, noun_predicted_note, K))

    print ("Using Proper Noun notes \n -----------------------------------------------------------------------------------\n")

    proper_noun_predicted_all = logistic_model_all.predict_proba(proper_noun_notes)[:, 1]
    proper_noun_predicted_note = logistic_model_note.predict_proba(proper_noun_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, proper_noun_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, proper_noun_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, proper_noun_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, proper_noun_predicted_note, K))

    print ("Using Adjective notes \n -----------------------------------------------------------------------------------\n")

    adjective_predicted_all = logistic_model_all.predict_proba(adjective_notes)[:, 1]
    adjective_predicted_note = logistic_model_note.predict_proba(adjective_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, adjective_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, adjective_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, adjective_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, adjective_predicted_note, K))

    print ("Using Verb notes \n -----------------------------------------------------------------------------------\n")

    verb_predicted_all = logistic_model_all.predict_proba(verb_notes)[:, 1]
    verb_predicted_note = logistic_model_note.predict_proba(verb_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, verb_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, verb_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, verb_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, verb_predicted_note, K))

    print ("\n\n --------------------------------------------------------------------------------------------------------\n")

    print ("Using Notes that are copy pasted\n -----------------------------------------------------------------------------------\n")

    copy_predicted_all = logistic_model_all.predict_proba(copy_notes)[:, 1]
    copy_predicted_note = logistic_model_note.predict_proba(copy_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, copy_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, copy_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, copy_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, copy_predicted_note, K))

    print ("Using Notes that are NOT copy pasted\n -----------------------------------------------------------------------------------\n")

    non_copy_predicted_all = logistic_model_all.predict_proba(non_copy_notes)[:, 1]
    non_copy_predicted_note = logistic_model_note.predict_proba(non_copy_notes)[:, 1]
    print ("ROC AUC Score for All features on Test Set:")
    print (roc_auc_score(test_y, non_copy_predicted_all))
    print ("ROC AUC Score for only Note features on Test Set:")
    print (roc_auc_score(test_y, non_copy_predicted_note))

    print ("Mortality @ K for All features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, non_copy_predicted_all, K))

    print ("Mortality @ K for only Note features on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(test_y, non_copy_predicted_note, K))

    print ("\n\n ------ DONE!!------------------------------------\n")
