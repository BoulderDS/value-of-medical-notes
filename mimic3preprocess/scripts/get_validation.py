import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='get readmission data that have physician notes')
parser.add_argument('--path', type=str,
                    default="/data/joe/physician_notes/mimic-data/",help='path to data root dir')
parser.add_argument('--task', type=str,
                    default="mortality", choices=["readmission", "mortality"], help='task type')
parser.add_argument('--period', type=str
                    , choices=["24", "48", "retro"], help='task type')
args = parser.parse_args()


listfile_dir = os.path.join(args.path, args.task)
train = pd.read_csv(os.path.join(listfile_dir, f"train_{args.period}_listfile.csv"))

print("Training data before split", train)
train_list = os.listdir(os.path.join(args.path,'train'))
if args.task == 'mortality':
    random_state = 20
else:
    random_state = 191
train_list, val_list = train_test_split(train_list, test_size =0.2, random_state=random_state)
train_list, val_list = set(train_list), set(val_list)
trains = []
vals = []
for row in train.values:
    if row[0].split("_")[0] in train_list:
        trains.append(row)
    else:
        vals.append(row)

trains = np.vstack(trains)
vals = np.vstack(vals)

trains = pd.DataFrame(trains, columns=['stay', 'y_true'])
vals = pd.DataFrame(vals, columns=['stay', 'y_true'])
count_train, count_val = np.sum(trains['y_true']), np.sum(vals['y_true'])
print(f"Training date size: {len(trains)}, {count_train/len(trains)}")
print(f"Validation date size: {len(vals)}, {count_val/len(vals)}")
trains.to_csv(os.path.join(listfile_dir, f'train_{args.period}_listfile_tmp.csv'), index=False)
vals.to_csv(os.path.join(listfile_dir, f'valid_{args.period}_listfile.csv'), index=False)
