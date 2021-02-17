import pandas as pd
import os
from nltk import word_tokenize
import numpy as np
import argparse
from multiprocessing import  Pool


note2id = {'Nursing/other': 900001, 'Physician': 900002, 'Nutrition': 900003, 'General': 900004, 'Nursing': 900005, 'Respiratory ': 900006,'Rehab Services': 900007, 'Social Work': 900008, 'Echo': 900010,'ECG': 900011,'Case Management ': 900012,'Pharmacy': 900013,'Consult': 900014, 'Radiology': 900015, 'Discharge summary': 900016}
columns =  ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015', '900016']

def count_token(inputs):
    stays_list, DATA_DIR = inputs
    # print(stays)
    feature_path = f'{DATA_DIR}/timeseries_features_retro/note_mask/'
    note_path =  f'{DATA_DIR}/timeseries_features_retro/note/'
    
    is_notes = []
    notes_count = np.zeros((len(columns),))
    notes_word_count = np.zeros((len(columns),))
    notes_word_len = {col:[] for col in columns}
    notes_len_full = []
    token_in_admissions = []
    stays = []


    for i, stay in enumerate(stays_list):
        # if i%1000 == 0:
        #     print(f"processed {i} admission")
        tmp_word_len = {col:[] for col in columns}
        note_mask = np.load(os.path.join(feature_path, stay), allow_pickle=True).astype(float)
        notes_num = note_mask.sum(axis=0)
        notes_count += notes_num
        notes = pd.read_csv(os.path.join(note_path, stay[:-4]+'.csv')).fillna("")
        stays.append(stay[:-4]+'.csv')
        token_in_admission = 0
        for i, col in enumerate(columns):
            is_notes = note_mask[:,i]
            for n, is_note in zip(notes[col], is_notes):
                if is_note:
                    n_len = len(word_tokenize(str(n)))
                    notes_word_count[i] += n_len
                    token_in_admission += n_len
                    notes_word_len[col].append(n_len)
                    tmp_word_len[col].append(n_len)
        notes_len_full.append(tmp_word_len)
        token_in_admissions.append(token_in_admission)
    
    return notes_len_full, token_in_admissions, stays

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, required=True)
    parser.add_argument('-n_workers', type=int, required=True)

    args = parser.parse_args()
    feature_path = f'{args.data_dir}/timeseries_features_retro/note_mask/'
    stays = os.listdir(feature_path)
    
    
    pool = Pool(args.n_workers)
    stay_splits = np.array_split(stays, args.n_workers)
    inputs = zip(stay_splits, [args.data_dir]*args.n_workers)
    outputs = pool.map(count_token, inputs) # n_workers * 2
    pool.close()
    pool.join()
    notes_len_full_n, token_in_admissions_n, stays_n = list(zip(*outputs)) # 2 * n_workers
    notes_len_full = [ n_len   for worker in notes_len_full_n for n_len in worker]
    token_in_admissions = [ n_token   for worker in token_in_admissions_n for n_token in worker]
    stays = [ stay   for worker in stays_n for stay in worker]

    print("Get note length" )
    df = pd.DataFrame({"stay":stays,"token_length":token_in_admissions})
    print(df)
    df.to_csv(f'{args.data_dir}/stay2token_all.csv')

    print("\nAll but discharge" )
    tokens = []
    for n in notes_len_full:
        tmp = 0
        # except discharge notes
        for col in ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015']:
            tmp += sum(n[col])
        tokens.append(tmp)
    df = pd.DataFrame({"stay":stays,"token_length":tokens})
    print(df)
    df.to_csv(f'{args.data_dir}/stay2token_all_but_discharge.csv')

    print("\ndischarge summaries")
    tokens = []
    for n in notes_len_full:
        tmp = 0
        # except discharge notes
        for col in ['900016']:
            if n[col]:
                tmp = n[col][-1]
        tokens.append(tmp)
    df = pd.DataFrame({"stay":stays,"token_length":tokens})
    df.to_csv(f'{args.data_dir}/stay2token_discharge.csv')
    print(df)
    
    print("\nlast physician notes")
    tokens = []
    count_one = 0
    for n in notes_len_full:
        tmp = 0
        # except discharge notes
        for col in ['900002']:
            if n[col]:
                tmp = n[col][-1]
            if len(n[col]) == 1:
                count_one += 1
        tokens.append(tmp)
    df = pd.DataFrame({"stay":stays,"token_length":tokens})
    df = df[df.token_length>0]
    df.to_csv(f'{args.data_dir}/stay2token_last_physician.csv')
    print(df)

if __name__ == '__main__':
    main()