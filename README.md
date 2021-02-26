# Characterizing the Value of Information in Medical Notes

This repository is the implementation of [Characterizing the Value of Information in Medical Notes](https://www.aclweb.org/anthology/2020.findings-emnlp.187/) at Findings of EMNLP2020.

# Table of Contents
1. [Environment](#1-environment)
2. [Preprocessing MIMIC-III](#2-preprocessing-mimic-iii)
3. [Train models with all information (structured/notes/structured+notes)](#3-train-models-with-all-information-(structured/notes/structured+notes))
4. [Note Type Comparison](#4-note-type-comparison)
5. [Note Portion Comparison](#5-note-portion-comparison)
6. [Note Portion Comparison Based on Length (Quartile analysis)](#6-note-portion-comparison-based-on-length-(quartile-analysis))
7. [Troubleshooting](#troubleshooting)
8. [Contact](#contact)


## 1. Environment

```
# open a new conda with python3.7
conda create -n notes python=3.7
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. Preprocessing MIMIC-III
First, you need to download [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) dataset as csv files to your machine.
Then change env variables in `scripts.mimic3preprocess.sh` for preprocessing.
```
export NUM_WORKER=32 # number of threads for multithreading
export DATA_DIR=/data/joe/physician_notes/mimic-data/ #path to your mimic csv files 
export OUTPUT_DIR=/data/joe/physician_notes/mimic-data/preprocessed/
```
Then, you may run `bash scripts.mimic3preprocess.sh`. This can take several hours (depends on number of workers) and up to ~600GB storage. Try to cut off unnecessary operations, e.g., 48 hours data, in `scripts.mimic3preprocess.sh` to save storage and computation time.

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

## 3. Train models with all information (structured/notes/structured+notes)
Note that we use all notes but discharge summaries for mortality prediction and all notes for readmission prediction.
After finishing following training, check performance here `notebooks/results_plots.ipynb`.
### Logistic regression
We first train logistic regression on two tasks. 
```
# 24hr Mortality prediction
bash scripts/logisitc_regression_mortality.sh

# readmission prediction
bash scripts/logisitc_regression_readmission.sh
```

### Deep Averaging Networks (DAN)
First, `cd models/DeepAverageNetwork` to working directory.
1. Build vocabulary 
Change ENV in `scripts/build_vocab.sh`.
```
# remember to run this command in DeepAverageNetwork dir
bash scripts/build_vocab.sh
```
2. Train models
```
bash scripts/run_text.sh 
bash scripts/run_feature.sh 
bash scripts/run_text_feature.sh 
```

## 4. Note Type Comparison
We need to build patient2notes table first. 
```
# at base dir. (~10 mins)
python -m processing.find_patient_with_sameNotes -data_dir /data/test_mimic_output/ -period 24
python -m processing.find_patient_with_sameNotes -data_dir /data/test_mimic_output/ -period retro
```
### Logistic Regression 
Change `DATA_DIR` in `scripts/logistic_regression_compare_notes_pairwise.sh` and run
```
bash scripts/logistic_regression_compare_notes_pairwise.sh
```
In this step, we will conduct pairwise comparison between every two types of note on admissions with these two types of note. To make a fair comparison, we downsampling note type having longer tokens to have the same number of tokens as its countpart. Also, we compute mean score over 10 experiment with different random seeds.

Once finishing the script, you can use notebook `notebooks/note_comparison_heatmap.ipynb` to visualize the note comparison with heatmap.

### Deep Averaging Networks
TODO: cleaning code

## 5. Note Portion Comparison
Change `DATA_DIR` in `scripts/sentence_select_similarity.sh` and `scripts/sentence_select.sh`. Then run
```
bash scripts/sentence_select.sh
bash scripts/sentence_select_similarity.sh
```
After finishing it, you can make plot in `notebooks/heurisitics_group_notes_plot-new.ipynb`.

### Logistic Regression
Change `model` in `scripts/sentence_select_similarity.sh` and `scripts/sentence_select.sh` to `LR`.

### Deep Averaging Networks
Change `model` in `scripts/sentence_select_similarity.sh` and `scripts/sentence_select.sh` to `DAN`.

## 6. Note Portion Comparison Based on Length (Quartile analysis)
We first need to count number of tokens in admissions.
```
python -m processing.count_token -data_dir PATH_TO_PROCESSED_DATA_DIR -n_worker 60
```
Then, we will split selected sentences into each quartile.
```
python -m processing.quartile_split -data_dir PATH_TO_PROCESSED_DATA_DIR
```
It's fine to have this error `FileNotFoundError: [Errno 2] No such file or directory: '/data/test_mimic_output//select_sentence/DAN/mortality'` if you have run previous step with DAN.
Finally, you can now visualize plots in `notebooks/heurisitics_group_notes_plot-new.ipynb`.

## Troubleshooting
1. `pandas` version might affect `python -m  mimic3preprocess.scripts.extract_subjects $DATA_DIR $OUTPUT_DIR`. Follow the env version if you have the same problem.

## Contact
Chao-Chun Hsu, chaochunh@uchicago.edu
```
@inproceedings{hsu2020characterizing,
  title={Characterizing the Value of Information in Medical Notes},
  author={Hsu, Chao-Chun and Karnwal, Shantanu and Mullainathan, Sendhil and Obermeyer, Ziad and Tan, Chenhao},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={2062--2072},
  year={2020}
}
```
