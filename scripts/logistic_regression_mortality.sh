#!/bin/bash
set -x

export DATA_DIR=/data/test_mimic_output/
export OUTPUT_DIR=/data/test_mimic_output/logistic_regression/

mkdir -p $OUTPUT_DIR

balanced="--balanced"
TASK="mortality"
mkdir -p $OUTPUT_DIR/results_$TASK/
#echo "Run logistic regression on structured data"
#for i in 24 48 retro
#do
#(python -m models.logistic_regression.logistic_regression_readmission --feature $i --task $TASK $balanced > /home/joe/physician_notes/models/logistic_regression/results_$TASK/$i.txt) &
#done

echo "Run LR on structured data that have physician_notes"
for i in 24 #48 retro
do
    for NOTE in all_but_discharge # physician_nursing  physician 
    do
        for f in all  all_but_notes notes
        do
            (python -m models.logistic_regression.logistic_regression --data $DATA_DIR --feature_period $i --feature_used $f $balanced --note $NOTE --task $TASK --metric pr $balanced > $OUTPUT_DIR/results_$TASK/$i\_$NOTE\_$f.txt)&
        done
    done
done
