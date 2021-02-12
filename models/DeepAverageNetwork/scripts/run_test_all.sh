#!/bin/bash
set -x


export DATA_DIR=/data/test_mimic_output/

for i in 24 # 48 retro
do
for n in  all_but_discharge # physician physician_nursing
do
    (python -m main -device 0 -name  $n -period $i -data_dir $DATA_DIR -feature -text -task mortality) 
done
done
#for i in physician discharge
for i in all # discharge
do
    (python -m main -device 0 -name $i -period retro $DATA_DIR -feature -text -task readmission)
done
