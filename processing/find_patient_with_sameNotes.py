import pandas as pd
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', type=str, required=True)
parser.add_argument('-period', type=str)
args = parser.parse_args()
periods = [args.period]
columns = ['900001','900002','900003','900004','900005','900006','900007','900008','900010','900011','900012','900013','900014','900015', '900016']
for period in periods:
    stays = []
    is_notes = []
    feature_path = f'{args.data_dir}/timeseries_features_{period}/note_mask/'
    for i, stay in enumerate(os.listdir(feature_path)):
        if i % 5000 == 0:
            print(f"processed {i} admission")
        note_mask = np.load(os.path.join(feature_path, stay), allow_pickle=True).astype(float)
        note_mask = 1-np.equal(note_mask.sum(axis=0), 0).astype(float)  # check existance of each
        stays.append(stay[:-4]+'.csv')
        is_notes.append(note_mask)
    df = pd.DataFrame(data=is_notes, columns=columns)
    df.insert(0, 'stay', stays)
    df.to_csv(f'{args.data_dir}/patient2notes_{period}.csv', index=False)
