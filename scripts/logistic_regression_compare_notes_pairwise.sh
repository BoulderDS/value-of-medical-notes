#!/bin/bash

set -x
export DATA_DIR=/data/test_mimic_output/

for i in 0 1 2 3 4 5 6 7 8 9
do
    (python -m models.logistic_regression.compare_note_pairwise --data $DATA_DIR --feature_period 24 --feature_used notes --task mortality --note all_but_discharge --segment $i)&
    (python -m models.logistic_regression.compare_note_pairwise --data $DATA_DIR --feature_period retro --feature_used notes --task readmission --note all --segment $i)&
done