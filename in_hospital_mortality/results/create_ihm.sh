#!/bin/bash

python -m mimic3preprocess.scripts.create_in_hospital_mortality /data/physician_notes/preprocessed/ /data/physician_notes/new_experiments/in_hospital_mortality_24/ 24
python -m mimic3preprocess.scripts.create_in_hospital_mortality /data/physician_notes/preprocessed/ /data/physician_notes/new_experiments/in_hospital_mortality_48/ 48
python -m mimic3preprocess.scripts.create_in_hospital_mortality /data/physician_notes/preprocessed/ /data/physician_notes/new_experiments/in_hospital_mortality_retro/ -1
