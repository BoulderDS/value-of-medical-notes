import numpy as np
import pandas as pd
import os, pickle
import argparse
parser = argparse.ArgumentParser(description='merge features into one file.')
parser.add_argument('data_path', type=str,
                    help='Directory of listfile.')
parser.add_argument('feature_path', type=str,
                    help='Directory of extracted features.')
parser.add_argument('note', type=str, choices=['physician', 'discharge'],
                    help='task type')
parser.add_argument('period', type=str, choices=['24', '48', 'retro'],
                    help='length of events')
args = parser.parse_args()

train_list = pd.read_csv(os.path.join(args.data_path, f"{args.note}_note_train_{args.period}.csv"))
features = pickle.load(open(args.feature_path, "rb"))

sum_value = np.zeros((766*7*6,), dtype=np.float)
min_value = np.ones((766*7*6,), dtype=np.float)*np.inf
max_value = np.ones((766*7*6,), dtype=np.float)*(-np.inf)
count = np.zeros((766*7*6,))

for i, stay in enumerate(train_list['stay']):
    if i+1 % 5000 == 0:
        print(f'processed {i} stays')
    feat = features[stay]
    mask = 1 - pd.isnull(feat).astype(int)
    feat = np.nan_to_num(feat.astype(float))
    # for minmax scaler
    max_value = np.maximum(max_value, feat)
    min_value = np.minimum(min_value, feat)

    # calculate mean
    sum_value += feat
    count += mask

max_value = np.nan_to_num(max_value, posinf=0., neginf=0.)
min_value = np.nan_to_num(min_value, posinf=0., neginf=0.)

print(sum_value.shape, count.shape)
x_mean = sum_value/count
print(x_mean)
pickle.dump(x_mean, open(args.feature_path[:-4]+"_all_x_mean.pkl", "wb"))
pickle.dump(max_value, open(args.feature_path[:-4]+"_all_x_max.pkl", "wb"))
pickle.dump(min_value, open(args.feature_path[:-4]+"_all_x_min.pkl", "wb"))
