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
from nltk.tokenize import TweetTokenizer
import models.sentence_select.utils as utils
nlp = TweetTokenizer()

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
    test_list = pd.read_csv(f"/data/joe/physician_notes/mimic-data/{args.task}/{args.note}_note_test_{args.feature_period}.csv")

    #print("Loading data")
    # features = pickle.load(open(featurepath,'rb'))
    note_ids = Config.note_type[args.note]

    results = {}
    for note_id in note_ids:
        X_test_notes, y_test = [], []
        for index, row in test_list.iterrows():
            note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
            note_collect = []
            tmp_from_notes = []

            notes = [str(v) for v in note[note_id] if str(v) != '']
            note_collect.extend(notes)
            tmp_from_notes.extend([note_id]*len(notes))

            if note_id in ["900001", "900005"]:
                note_tmp_id = "900001" if note_id == "900005" else "900005"
                notes = [str(v) for v in note[note_tmp_id] if str(v) != '']
                note_collect.extend(notes)
                tmp_from_notes.extend([note_tmp_id]*len(notes))
            
            if args.segment:
                sentences, tmp_from_notes = utils.segmentSentence(model, args.segment, note_collect, tmp_from_notes, datapath, row['stay'], compare=True)
            else:
                sentences = [" ".join(note_collect)]
            X_test_notes.extend(sentences)
            y_test.extend([row['y_true']]*len(sentences))
        
        test_notes = pd.DataFrame({'file_name': test_list['stay'].values, 'text': X_test_notes})
        res = model.predict_proba(test_notes)[:,1]
        results[note_id] = res

    test_file = pd.read_csv(f"/data/joe/physician_notes/mimic-data/{args.task}/{args.note}_note_test_{args.feature_period}.csv")
    df = pd.DataFrame(results)
    df.insert(0,'stay', test_file['stay'])
    if not os.path.exists('/home/joe/physician_notes/models/logistic_regression/compare_notes/'):
        os.mkdir('/home/joe/physician_notes/models/logistic_regression/compare_notes/')
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

    df.to_csv(f'/home/joe/physician_notes/models/logistic_regression/compare_notes/{model_name}', index=False)

