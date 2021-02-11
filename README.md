# physician_notes

## 1. Environment

```
python -m spacy download en_core_web_sm
```

## 2. Preprocessing MIMIC-III
First, you need to download MIMIC-III dataset as csv files into your machine.
Then change env variables in `scripts.mimic3preprocess.sh` for preprocessing.
```
export NUM_WORKER=32 # number of threads for multithreading
export DATA_DIR=/data/joe/physician_notes/mimic-data/ #path to your mimic csv files 
export OUTPUT_DIR=/data/joe/physician_notes/mimic-data/preprocessed/
```
Then, you may run `bash scripts.mimic3preprocess.sh`. This can take several hours (depends on number of workers) and up to ~500GB storage. Try to cut off unnecessary operations, e.g., 48 hours data, in `scripts.mimic3preprocess.sh` to save storage and computation time.

We briefly introduce the functionality of every script as follows:
1. `mimic3preprocess.scripts.extract_subjects`: group records by patient id.
2. `mimic3preprocess.scripts.split_train_and_test`: move patient to train/test split based on a given list in `mimic3preprocess/resources/testset.csv`.
3. `mimic3preprocess.scripts.extract_episodes_from_subjects_multiprocessing`: split admissions of a patient into different episodes.
4. `mimic3preprocess.scripts.feature_extraction_multiprocessing`: we convert structured variables into a fixed-length vector with six statistical function and normalization. 
5. `mimic3preprocess.scripts.merge_features`: we merge structured variables vectors of all patients into a big dictionary (look-up table) to speed up training process.
6. `mimic3preprocess.scripts.timeseries_feature_extraction_multiprocessing`: preprocess data into timeseries of structured variables and notes. Also, we add necessary components to train GRU-D model.
7. `mimic3preprocess.scripts.create_in_hospital_mortality` and `mimic3preprocess.scripts.create_readmission`: create (in-hospital moratlity prediction task) or (readmission prediction task) given a period of records (24hrs/48hrs/retrospective (all)). To align with the main paper, we only process mortality 24 hrs and readmission retro as defult. 
8. `mimic3preprocess.scripts.get_validation`: split data from previous step into train/val/test.
9. `mimic3preprocess.scripts.create_in_hospital_mortality_note`: find admissions that contains a specific set of note types.
9. `mimic3preprocess.scripts.get_data_with_notes`: filter out admissions that have no specific set of note in previous step.

After preprocessing, yor should get train/valid/test set as the following files.
```
# naming format '{args.note}_note_test_{args.period}.csv'
# {note} is for note types used and {period} is the time span of records

# for 24 hrs mortality prediction
OUTPUT_DIR/mortality/all_but_discharge_note_{train/valid/test}_24.csv

# for retro readmission prediction  
OUTPUT_DIR/readmission/all_note_{train/valid/test}_retro.csv
```

## Troubleshooting
1. `pandas` version might affect `python -m  mimic3preprocess.scripts.extract_subjects $DATA_DIR $OUTPUT_DIR`. Follow the env version if you have the same problem.