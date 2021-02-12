#!/usr/bin/env python3
from pprint import pprint
from time import time
import pickle
import models.config as Config
from in_hospital_mortality.feature_definitions import BOWFeatures, DictFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler #, StandardScalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize
import models.sentence_select.utils as utils


def segmentSample(note_1, note_2, from_notes_1, from_notes_2,  is_token=False, is_percent=False):
    sentences_1 = []
    note_ids_1 = []
    sentences_2 = []
    note_ids_2 = []

    for n, n_id in zip(note_1, from_notes_1):
        tmp = sent_tokenize(n)
        sentences_1.extend(tmp)
        note_ids_1.extend([n_id]*len(tmp))
    for n, n_id in zip(note_2, from_notes_2):
        tmp = sent_tokenize(n)
        sentences_2.extend(tmp)
        note_ids_2.extend([n_id]*len(tmp))

    lengths_1 = [len(word_tokenize(s)) for s in sentences_1]
    lengths_2 = [len(word_tokenize(s)) for s in sentences_2]
    len_1 = sum(lengths_1)
    len_2 = sum(lengths_2)
    min_len = min(len_1, len_2)
    # if from_notes_2[0]=="900002":
    #     print(len_1, len_2)
    #     print(note_1)
    if len_1 == min_len:
        sent_1 = " ".join(sentences_1)
    else:
        idxs_1 = np.arange(len(lengths_1))
        np.random.shuffle(idxs_1)
        sent_1, _ = utils.get_tokens(sentences_1,min_len,lengths_1,idxs_1,note_ids_1, is_percent=is_percent)
    if len_2 == min_len:
        sent_2 = " ".join(sentences_2)
    else:
        idxs_2 = np.arange(len(lengths_2))
        np.random.shuffle(idxs_2)
        sent_2, _ = utils.get_tokens(sentences_2,min_len,lengths_2,idxs_2,note_ids_2, is_percent=is_percent)

    # if len(sentences) == 0:
    #     return [""],["0"]
    # idxs = np.arange(len(lengths))
    # np.random.shuffle(idxs)
    # bestsent, best_note_ids = utils.get_tokens(sentences,num,lengths,idxs,note_ids, is_percent=is_percent)

    
    return [sent_1], [sent_2]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default='/data/joe/physician_notes/mimic-data/preprocessed/')
    parser.add_argument('--feature_period', type=str, help='feature period',
                        choices=["24", "48", "retro"])
    parser.add_argument('--feature_used', type=str, help='feature used',
                        choices=["all", "notes", "all_but_notes"])
    parser.add_argument('--note', type=str, help='feature used',
                        choices=["physician", "physician_nursing", "discharge", "all", "all_but_discharge"])
    parser.add_argument('--task', type=str, help='task',
                        choices=["mortality", "readmission"])
    parser.add_argument('--text_length', type=int, help='text length', default=None)
    parser.add_argument('--segment', type=str, help='heuristics', default=None)
    args = parser.parse_args()
    print (args)

    np.random.seed(int(args.segment))

    model_name = args.note+"_"+args.feature_period + '.chkpt'
    if args.feature_used == "all":
        model_name = "feature_text_" + model_name
    elif args.feature_used == "all_but_notes":
        model_name = "feature_" + model_name
    else:
        model_name = "text_" + model_name
    path = f'/data/joe/physician_notes/logistic_regression/models/{args.task}/{model_name}'
    model = pickle.load(open(path, 'rb'))

    datapath = os.path.join(args.data, f"timeseries_features_{args.feature_period}")
    featurepath = f'/data/joe/physician_notes/mimic-data/preprocessed/features_{args.feature_period}.pkl'

    #print("Loading data")
    # features = pickle.load(open(featurepath,'rb'))
    note_ids = Config.note_type[args.note]
    note2id = {'Nursing/other': 900001, 'Physician': 900002, 'Nutrition': 900003, 'General': 900004, 'Nursing': 900005, 'Respiratory ': 900006,'Rehab Services': 900007, 'Social Work': 900008, 'Echo': 900010,'ECG': 900011,'Case Management ': 900012,'Pharmacy': 900013,'Consult': 900014, 'Radiology': 900015, 'Discharge summary': 900016}

    test_file = pd.read_csv(f"/data/joe/physician_notes/mimic-data/{args.task}/{args.note}_note_test_{args.feature_period}.csv")
    patient2notes = pd.read_csv(f"/data/joe/physician_notes/mimic-data/preprocessed/patient2notes_{args.feature_period}.csv")
    patient2notes['900001'] = patient2notes['900001'] + patient2notes['900005'] - (patient2notes['900001'] * patient2notes['900005'])

    if args.task == "readmission":
        notes_name = ["Nursing/other", "Radiology", "ECG", "Physician","Discharge summary"] # no ECG this time
    else:
        notes_name = ["Nursing/other", "Radiology", "ECG", "Physician"]

    results = {}
    for i in range(len(notes_name)):
        for j in range(i   , len(notes_name)):
            note_1 = notes_name[i]
            note_2 = notes_name[j]
            if note_1 == note_2: continue
            
            note_id_1 = str(note2id[note_1])
            note_id_2 = str(note2id[note_2])
            df = patient2notes[(patient2notes[str(note_id_1)]==1) & (patient2notes[str(note_id_2)]==1)] # get patients having both notes
            tmp_test = test_file[test_file['stay'].isin(df['stay'])]
            print(note_id_1, note_id_2, len(tmp_test))
            X_test_notes_1, y_test, X_test_notes_2 = [], [],  []
            for index, row in tmp_test.iterrows():
                note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
                note_collect_1 = []
                tmp_from_notes_1 = []
                note_collect_2 = []
                tmp_from_notes_2 = []
                
                # extract note 1
                notes = [str(v) for v in note[note_id_1] if str(v) != '']
                note_collect_1.extend(notes)
                tmp_from_notes_1.extend([note_id_1]*len(notes))                
                if note_id_1 in ["900001", "900005"]:
                    note_tmp_id = "900001" if note_id_1 == "900005" else "900005"
                    notes = [str(v) for v in note[note_tmp_id] if str(v) != '']
                    note_collect_1.extend(notes)
                    tmp_from_notes_1.extend([note_tmp_id]*len(notes))

                # extract note 2
                notes = [str(v) for v in note[note_id_2] if str(v) != '']
                note_collect_2.extend(notes)
                tmp_from_notes_2.extend([note_id_2]*len(notes))


                sentences_1, sentences_2 = segmentSample(note_collect_1, note_collect_2, tmp_from_notes_1, tmp_from_notes_2, is_token=True)
                X_test_notes_1.extend(sentences_1)
                X_test_notes_2.extend(sentences_2)
                y_test.extend([row['y_true']]*len(sentences_2))
            
            test_notes = pd.DataFrame({'file_name': tmp_test['stay'].values, 'text': X_test_notes_1})
            res_1 = model.predict_proba(test_notes)[:,1]
            pr_1 = average_precision_score(y_test, res_1)
            test_notes = pd.DataFrame({'file_name': tmp_test['stay'].values, 'text': X_test_notes_2})
            res_2 = model.predict_proba(test_notes)[:,1]
            pr_2 = average_precision_score(y_test, res_2)
            results[note_id_1+"_"+note_id_2] = [pr_1,pr_2]
    
    df = pd.DataFrame(results)
    print(df)
    if not os.path.exists('/home/joe/physician_notes/models/logistic_regression/compare_notes_pairwise/'):
        os.mkdir('/home/joe/physician_notes/models/logistic_regression/compare_notes_pairwise/')
    model_name = args.task+'_'+args.note +'_'+ args.feature_period + '.csv'
    if args.feature_used == "all":
        model_name = "feature_text_" + model_name
    elif args.feature_used == "all_but_notes":
        model_name = "feature_" + model_name
    else:
        model_name = "text_" + model_name
    if args.text_length:
        model_name = str(args.text_length) + "_" + model_name
    if args.segment:
        model_name = args.segment+ "_" + model_name

    df.to_csv(f'/home/joe/physician_notes/models/logistic_regression/compare_notes_pairwise/{model_name}', index=False)

