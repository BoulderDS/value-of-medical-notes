#!/usr/bin/env python3
from pprint import pprint
from time import time
import pickle
import models.config as Config
from in_hospital_mortality.custom_metrics import mortality_rate_at_k, train_val_compute
from in_hospital_mortality.feature_definitions import BOWFeatures, DictFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler #, StandardScalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import argparse

def precision_at_k(y_label, y_pred, k):
    rank = list(zip(y_label, y_pred))
    rank.sort(key=lambda x: x[1], reverse=True)
    num_k = len(y_label)*k//100
    return sum(rank[i][0] == 1 for i in range(num_k))/float(num_k)

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
    parser.add_argument('--balanced', dest="balanced", action="store_true", help = 'whether to use balanced class weights')
    parser.add_argument('--metric', default="roc", type=str, choices=["roc", "pr"], help = 'metrics')
    args = parser.parse_args()
    print (args)

    datapath = os.path.join(args.data, f"timeseries_features_{args.feature_period}")
    featurepath = f'{args.data}/features_{args.feature_period}.pkl'
    train_list = pd.read_csv(f"{args.data}/{args.task}/{args.note}_note_train_{args.feature_period}.csv")
    val_list = pd.read_csv(f"{args.data}/{args.task}/{args.note}_note_valid_{args.feature_period}.csv")
    test_list = pd.read_csv(f"{args.data}/{args.task}/{args.note}_note_test_{args.feature_period}.csv")

    metric = roc_auc_score if  args.metric == "roc" else average_precision_score
    print("Loading data")
    X_train_notes, y_train, X_val_notes, y_val, X_test_notes, y_test = [], [], [], [], [], []
    t=time()
    features = pickle.load(open(featurepath,'rb'))
    note_ids = Config.note_type[args.note]
    for index, row in train_list.iterrows():
        note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
        note_collect = []
        for note_id in note_ids:
            #if len(X_train_notes) > 6000:
            #    break
            note_collect.extend([str(v) for v in note[note_id].values])
        note = " ".join(note_collect)
        X_train_notes.append(note)
        y_train.append(row['y_true'])
    for index, row in val_list.iterrows():
        note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
        note_collect = []
        for note_id in note_ids:
            #if len(X_train_notes) > 6000:
            #    break
            note_collect.extend([str(v) for v in note[note_id].values])
        note = " ".join(note_collect)
        X_val_notes.append(note)
        y_val.append(row['y_true'])

    for index, row in test_list.iterrows():
        note = pd.read_csv(os.path.join(datapath, "note", row['stay']), dtype=object).fillna("")
        note_collect = []
        for note_id in note_ids:
            #if len(X_train_notes) > 6000:
            #    break
            note_collect.extend([str(v) for v in note[note_id].values])
        note = " ".join(note_collect)
        X_test_notes.append(note)
        y_test.append(row['y_true'])


    train_notes = pd.DataFrame({'file_name': train_list['stay'].values, 'text': X_train_notes})
    val_notes = pd.DataFrame({'file_name': val_list['stay'].values, 'text': X_val_notes})
    test_notes = pd.DataFrame({'file_name': test_list['stay'].values, 'text': X_test_notes})
    """
    train_notes = pd.DataFrame({'file_name': ["0"]*len(X_train_notes), 'text': X_train_notes})
    val_notes = pd.DataFrame({'file_name': ["0"]*len(X_val_notes) , 'text': X_val_notes})
    test_notes = pd.DataFrame({'file_name':  ["0"]*len(X_test_notes), 'text': X_test_notes})
    """
    union_list = []
    if args.feature_used in ['all', 'notes']:
        print ("add Bag of Words features .....")
        union_list.append(("tfidf_pipe",
                            Pipeline([
                            ("tfidf", BOWFeatures()),
                            ("scaler", MaxAbsScaler()),
                            ])))
    if args.feature_used in ['all','all_but_notes']:
        print ("add structured variable features ..... ")
        union_list.append(("structured",
                           Pipeline([
                               ("fe", DictFeatures(features)),
                               ("imputer", SimpleImputer()),
                               ("scaler", MinMaxScaler()),
                           ])))

    print("Total number of training data:", len(X_train_notes))
    print("Total number of validation data:", len(X_val_notes))
    print("Total number of test data:", len(X_test_notes))
    print("data loading time:", time()-t)

    pipeline = Pipeline([
        ('union', FeatureUnion(union_list)),
        ('lr', LogisticRegression(solver="lbfgs", max_iter = 500,
                                  class_weight="balanced" if args.balanced else None)),
    ])

    parameters = {
        "lr__C": np.logspace(-11, 0, 12, base = 2)
    }

    # Display of parameters

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    # Training on training data and hyperparameter tuning on validation data

    t0 = time()
    pipeline, best_score, best_parameters, params, scores = train_val_compute(train_notes, val_notes, y_train, y_val, pipeline, parameters, func=metric)
    print("done in %0.3fs" % (time() - t0))
    print()

    # Displaying training results

    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Best models")
    print(pipeline)
    print ("Mean test score:")
    print(scores)
    print("Best score: \n%0.3f" % best_score)
    val_predicted = pipeline.predict_proba(val_notes)[:, 1]
    print ("ROC AUC Score on val Set:")
    print(roc_auc_score(y_val, val_predicted))
    val_pr = best_score
    val_roc = roc_auc_score(y_val, val_predicted)


    # Displaying test results

    test_predicted = pipeline.predict_proba(test_notes)[:, 1]
    print ("ROC AUC Score on Test Set:")
    print(roc_auc_score(y_test, test_predicted))
    print ("PR AUC Score on Test Set:")
    print(average_precision_score(y_test, test_predicted))
    test_roc = roc_auc_score(y_test, test_predicted)
    test_pr = average_precision_score(y_test, test_predicted)
    print("save model")

    model_name = args.note +'_'+ args.feature_period + '.chkpt'
    if args.feature_used == "all":
        model_name = "feature_text_" + model_name
    elif args.feature_used == "all_but_notes":
        model_name = "feature_" + model_name
    else:
        model_name = "text_" + model_name
    path_dir = f'{args.data}/logistic_regression/models/{args.task}/'
    import pathlib
    pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(pipeline, open(os.path.join(path_dir, model_name), 'wb'))

    # save result
    result_dir = "./models/logistic_regression/results/"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(result_dir+f"{args.task}"):
        os.mkdir(result_dir+f"{args.task}")
    if not os.path.exists(result_dir+f"{args.task}/{args.note}"):
        os.mkdir(result_dir+f"{args.task}/{args.note}")


    val_predicted = pipeline.predict_proba(val_notes)[:, 1]
    val_ap = average_precision_score(y_val, val_predicted)
    val_roc = roc_auc_score(y_val, val_predicted)
    precision, recall, thresholds = precision_recall_curve(y_val, val_predicted)
    val_pr = auc(recall, precision)
    val_p_at_1 = precision_at_k(y_val, val_predicted, 1)
    val_p_at_5 = precision_at_k(y_val, val_predicted, 5)
    val_p_at_10 = precision_at_k(y_val, val_predicted, 10)

    test_predicted = pipeline.predict_proba(test_notes)[:, 1]
    test_ap = average_precision_score(y_test, test_predicted)
    test_roc = roc_auc_score(y_test, test_predicted)
    precision, recall, thresholds = precision_recall_curve(y_test, test_predicted)
    test_pr = auc(recall, precision)
    test_p_at_1 = precision_at_k(y_test, test_predicted, 1)
    test_p_at_5 = precision_at_k(y_test, test_predicted, 5)
    test_p_at_10 = precision_at_k(y_test, test_predicted, 10)

    # save result
    result_dir = "./models/logistic_regression/results/"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(result_dir+f"{args.task}"):
        os.mkdir(result_dir+f"{args.task}")
    if not os.path.exists(result_dir+f"{args.task}/{args.note}"):
        os.mkdir(result_dir+f"{args.task}/{args.note}")

    outname = f'{args.feature_period}.csv'
    if args.feature_used == "all":
        outname = "feature_text_" + outname
    elif args.feature_used == "notes":
        outname = "text_" + outname
    else:
        outname = "feature_" + outname

    print("Write Result to ", outname)
    with open(os.path.join(result_dir, args.task, args.note, outname), 'w') as f:
        f.write("TYPE,ROCAUC,PRAUC,AP,P@1,P@5,P@10\n")
        f.write(f"valid,{val_roc},{val_pr},{val_ap},{val_p_at_1},{val_p_at_5},{val_p_at_10}\n")
        f.write(f"test,{test_roc},{test_pr},{test_ap},{test_p_at_1},{test_p_at_5},{test_p_at_10}\n")

    if args.feature_used == "notes":
        tfidf_words = dict(pipeline.named_steps['union']
                        .transformer_list).get('tfidf_pipe').named_steps['tfidf'].get_feature_names()
        lr_coefs_pos = pipeline.named_steps['lr'].coef_[0].argsort()[::-1][:10]
        print(lr_coefs_pos)
        lr_coefs_neg = pipeline.named_steps['lr'].coef_[0].argsort()[:10]
        print("important pos words")
        for i in lr_coefs_pos:
            print(tfidf_words[i], pipeline.named_steps['lr'].coef_[0][i])
        print("important neg words")
        for i in lr_coefs_neg:
            print(tfidf_words[i], pipeline.named_steps['lr'].coef_[0][i])


    """
    print ("Mortality @ K on Test Set:")
    for K in [10, 50, 100, 500, 1000]:
        print ("K = {}".format(K))
        print (mortality_rate_at_k(y_val, val_predicted, K))
    """