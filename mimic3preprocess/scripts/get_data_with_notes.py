import pandas as pd
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description='get readmission data that have physician notes')
parser.add_argument('--path', type=str,
                    default="/data/joe/physician_notes/mimic-data/",help='path to data root dir')
parser.add_argument('--task', type=str,
                    default="mortality", choices=["readmission", "mortality"], help='task type')
parser.add_argument('--note', type=str,
                    default="physician", choices=["physician_nursing","physician", "discharge", "all", "all_but_discharge"], help='note type')
parser.add_argument('--period', type=str,
                    default="24", choices=["24", "48", "retro"],help='note period')
args = parser.parse_args()

listfile_dir = os.path.join(args.path, args.task)
train_listfile = pd.read_csv(os.path.join(listfile_dir, f"train_{args.period}_listfile_tmp.csv"))
valid_listfile = pd.read_csv(os.path.join(listfile_dir, f"valid_{args.period}_listfile.csv"))
test_listfile = pd.read_csv(os.path.join(listfile_dir, f"test_{args.period}_listfile.csv"))
print("train length:", len(train_listfile))
print("valid length:", len(valid_listfile))
print("test length:", len(test_listfile))



physician_train_mortality = pd.read_csv(f"/data/joe/physician_notes/mimic-data/{args.task}/train_{args.note}_{args.period}_listfile.csv")
physician_test_mortality = pd.read_csv(f"/data/joe/physician_notes/mimic-data/{args.task}/test_{args.note}_{args.period}_listfile.csv")
physician_mortality = physician_train_mortality.append(physician_test_mortality, ignore_index = True)

train_listfile = train_listfile[train_listfile['stay'].isin(physician_mortality['stay'])]
valid_listfile = valid_listfile[valid_listfile['stay'].isin(physician_mortality['stay'])]
test_listfile = test_listfile[test_listfile['stay'].isin(physician_mortality['stay'])]
print("train length:", len(train_listfile))
print("valid length:", len(valid_listfile))
print("test length:", len(test_listfile))
count_train, count_val = np.sum(train_listfile['y_true']), np.sum(valid_listfile['y_true'])
print(f"Training date size: {len(train_listfile)}, {count_train/len(train_listfile)}")
print(f"Validation date size: {len(valid_listfile)}, {count_val/len(valid_listfile)}")
train_listfile.to_csv(os.path.join(listfile_dir, f'{args.note}_note_train_{args.period}.csv'), index=False)
valid_listfile.to_csv(os.path.join(listfile_dir, f'{args.note}_note_valid_{args.period}.csv'), index=False)
test_listfile.to_csv(os.path.join(listfile_dir, f'{args.note}_note_test_{args.period}.csv'),  index=False)


