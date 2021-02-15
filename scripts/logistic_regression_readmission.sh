#!/bin/bash
set -x
export DATA_DIR=/data/test_mimic_output/
export OUTPUT_DIR=/data/test_mimic_output/logistic_regression/

mkdir -p $OUTPUT_DIR

balanced=""
TASK="readmission"
mkdir -p $OUTPUT_DIR/results_$TASK/

#echo "Run logistic regression on structured data"
#(python -m models.logistic_regression.logistic_regression_readmission --feature retro --task readmission $balanced > /home/joe/physician_notes/models/logistic_regression/results_readmission/retro.txt) &

echo "Run LR on structured data that have physician_notes"
for i in retro
do
    for NOTE in all # discharge # physician
    do
        for f in all all_but_notes notes
        do
            (python -m models.logistic_regression.logistic_regression --data $DATA_DIR --feature_period $i --feature_used $f $balanced --note $NOTE --metric pr --task readmission >  $OUTPUT_DIR/results_$TASK/$i\_$NOTE\_$f.txt) &
        done
    done
done
