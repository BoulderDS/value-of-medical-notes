from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import argparse
import numpy as np
import os
import pathlib


def precision_at_k(y_label, y_pred, k):
    # precision @ K percent
    rank = list(zip(y_label, y_pred))
    rank.sort(key=lambda x: x[1], reverse=True)
    num_k = len(y_label)*k//100
    num_k = 1 if num_k < 1 else num_k
    return sum(rank[i][0] == 1 for i in range(num_k))/float(num_k)


def get_quartile(df):
    x = df['token_length'].values
    q1 = np.quantile(x, 0.25)
    q2 = np.quantile(x, 0.50)
    q3 = np.quantile(x, 0.75)
    return [(0, q1), (q1, q2), (q2, q3), (q3, float('inf'))]


def get_df_len(task, heu, res):

    if "discharge" == res.split("_")[0]:
        return pd.read_csv(f'{DATA_DIR}/stay2token_discharge.csv')
    elif "physician" == res.split("_")[0]:
        return pd.read_csv(f'{DATA_DIR}/stay2token_last_physician.csv')
    else:
        if task == "readmission":
            return pd.read_csv(f'{DATA_DIR}/stay2token_all.csv')
        elif task == "mortality":
            return pd.read_csv(f'{DATA_DIR}/stay2token_all_but_discharge.csv')


def get_group_result(df):
    if len(df) == 0:
        df = pd.DataFrame(
            data=[["test", 0, 0, 0, 0, 0, 0]],
            columns=["TYPE", "ROCAUC", "PRAUC", "P@1", "P@5", "P@10", "NUM_ADMIT"])
        return df
    label, pred = df['y_label'], df['prob']
    p_at_1 = precision_at_k(label, pred, 1)
    p_at_5 = precision_at_k(label, pred, 5)
    p_at_10 = precision_at_k(label, pred, 10)
    pr = average_precision_score(label, pred)
    try:
        roc = roc_auc_score(label, pred)
    except ValueError:
        roc = -1
    df = pd.DataFrame(
        data=[
            ["test", roc, pr, p_at_1, p_at_5, p_at_10, len(df)]
        ],
        columns=["TYPE", "ROCAUC", "PRAUC", "P@1", "P@5", "P@10", "NUM_ADMIT"]
    )
    
    return df


parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', type=str, required=True)
args = parser.parse_args()
DATA_DIR = args.data_dir

for model in [ 'LR','DAN']: #'LR',
    for task in ['mortality', 'readmission']:
        for i, heu in enumerate(os.listdir(f'{DATA_DIR}/select_sentence/{model}/{task}')):
            if (i+1)%10 == 0: print(i,"task")
            if "csv" in heu: continue
            if "@" in heu :continue
            if "train" in heu: continue
            if "valid" in heu: continue
            for res in os.listdir(f'{DATA_DIR}/select_sentence/{model}/{task}/{heu}/'):
                print(f'{DATA_DIR}/select_sentence/{model}/{task}/{heu}/{res}')
                df_len = get_df_len(task, heu, res) 
                df = pd.read_csv(f'{DATA_DIR}/select_sentence/{model}/{task}/{heu}/{res}', lineterminator='\n')
                df = pd.merge(df, df_len, on="stay")
                quartile_set = get_quartile(df)
                for q, (g_low, g_high) in enumerate(quartile_set):
                    import pathlib
                    path = f'{DATA_DIR}/select_sentence/{model}/group/q{q}/{task}/{heu}/'
                    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

                    result = get_group_result(df[(df['token_length']>=g_low) & (df['token_length']<g_high)])
                    result.to_csv(f'{DATA_DIR}/select_sentence/{model}/group/q{q}/{task}/{heu}/{res}')

notes = ['all']  # ', 'discharge', 'last_physician', 'last_nursing'
test_notes = ['test_token_percent_1/', 'test_token_percent_1/'] #, 'sim_token_100/', 'sim_token_100/physician_', 'sim_token_100/nursing_'
for model in ['LR', 'DAN']:  # 'LR',
    for task in ['readmission']:  # 'mortality', 
        for i, res in enumerate(os.listdir(f'{DATA_DIR}/select_sentence/{model}/{task}/all')): 
            print(f'{DATA_DIR}/select_sentence/{model}/{task}/all/{res}')
            period = res.split(".")[0].split("_")[-1]
            for n, t_n in zip(notes, test_notes):
                try:
                    test_list = pd.read_csv(f'{DATA_DIR}/select_sentence/{model}/{task}/longest_{t_n}{res}')
                    df_len = pd.read_csv(f'{DATA_DIR}/stay2token_{n}.csv')
                    df_len = df_len[df_len['token_length']>0]
                    df = pd.read_csv(f'{DATA_DIR}/select_sentence/{model}/{task}/all/{res}', lineterminator='\n')
                    df = df[df['stay'].isin(test_list['stay'])]
                    df = pd.merge(df, df_len, on="stay")
                    print(n,len(df))
                    quartile_set = get_quartile(df)
                    print(quartile_set)
                    if n in ['last_physician', 'discharge']:
                        print(n, quartile_set)
                    for q, (g_low, g_high) in enumerate(quartile_set):
                        path = f'{DATA_DIR}/select_sentence/{model}/group/q{q}/{task}/all/'
                        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                        result = get_group_result(df[(df['token_length'] >= g_low) & (df['token_length'] < g_high)])
                        idx = t_n.index("/")
                        print("t n:", t_n)
                        result.to_csv( f'{DATA_DIR}/select_sentence/{model}/group/q{q}/{task}/all/{t_n[idx:]}{res}')
                        print(f'{DATA_DIR}/select_sentence/{model}/group/q{q}/{task}/all/{t_n[idx:]}{res}')

                except:
                    print("error", n)