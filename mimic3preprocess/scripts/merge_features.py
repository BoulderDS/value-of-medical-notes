import numpy as np
import pandas as pd
import os, pickle
import argparse
parser = argparse.ArgumentParser(description='merge features into one file.')
parser.add_argument('data_path', type=str,
                    help='Directory of listfile.')
parser.add_argument('feature_path', type=str,
                    help='Directory of extracted features.')
args = parser.parse_args()

train_list = pd.read_csv(os.path.join(args.data_path, "train", "listfile.csv"))
test_list = pd.read_csv(os.path.join(args.data_path, "test", "listfile.csv"))
all_files = list(train_list['stay']) + list(test_list['stay'])

output = {}
for index, row in train_list.iterrows():
    feat = np.load(args.feature_path+row['stay'][:-3]+'npy')
    output[row['stay']] = feat

for index, row in test_list.iterrows():
    feat = np.load(args.feature_path+row['stay'][:-3]+'npy')
    output[row['stay']] = feat

pickle.dump(output, open(args.feature_path[:-1]+".pkl", "wb"))
