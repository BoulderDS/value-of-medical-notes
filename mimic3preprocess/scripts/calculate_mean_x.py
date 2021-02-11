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
parser.add_argument('feature_path', type=str,
                    help='Directory of extracted features.')

sum_value = np.zeros((766,), dtype=np.float)
min_value = np.ones((766,), dtype=np.float)*np.inf
max_value = np.ones((766,), dtype=np.float)*(-np.inf)
count = np.zeros((766,))

for i, stay in enumerate(train_list['stay']):
    if i+1 % 5000 == 0:
        print(f'processed {i} stays')
    feat = np.load(os.path.join(args.feature_path, "structured_data", stay[:-4]+".npy"), allow_pickle=True)
    mask = np.load(os.path.join(args.feature_path, "structured_mask", stay[:-4]+".npy"), allow_pickle=True)
    feat = np.nan_to_num(feat.astype(float))

    # for minmax scaler
    if feat.shape[0] == 0:
        continue
    feat_max = np.amax(feat, axis=0)
    feat_min = np.amin(feat, axis=0)

    max_value = np.maximum(max_value, feat_max)
    min_value = np.minimum(min_value, feat_min)

    # calculate mean
    sum_value += np.sum(feat, axis=0)
    count += np.sum(mask, axis=0)

max_value = np.nan_to_num(max_value, posinf=0., neginf=0.)
min_value = np.nan_to_num(min_value, posinf=0., neginf=0.)

print(sum_value.shape, count.shape)
x_mean = sum_value/count
print(x_mean)
pickle.dump(x_mean, open(args.feature_path[:-1]+"_x_mean.pkl", "wb"))
pickle.dump(max_value, open(args.feature_path[:-1]+"_x_max.pkl", "wb"))
pickle.dump(min_value, open(args.feature_path[:-1]+"_x_min.pkl", "wb"))
