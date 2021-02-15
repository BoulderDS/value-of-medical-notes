#!/bin/bash
set -x
feature="notes"
split="test"
model="LR" # or "DAN" 
export DATA_DIR=/data/test_mimic_output/

for t in  fracmedicalsent_norm_$split longest_$split   # fracmedicalsent_$split # countmedicalsent # fracmedicalsent longestwindow fracmedical window  fracmedical   countmedical  #section 
do
    for m in  $model  #  DAN 
    do
        for s in  $t\_token\_percent\_15 $t\_token\_percent\_25 $t\_token\_percent\_30 $t\_token\_percent\_35 $t\_token\_percent\_40 $t\_token\_percent\_45  $t\_token\_percent\_1 $t\_token\_percent\_5 $t\_token\_percent\_10 $t\_token\_percent\_20  $t\_token\_percent\_50 #$t\_token\_100  $t\_token\_200  $t\_token\_400 # $t\_token\_800   $t\_token\_1600   $t\_token\_1000 $t\_token\_2000   $t\_1 $t\_5 $t\_10 $t\_20  $t\_50
        do

            for p in 24 # 48 retro
            do
                (python -m models.sentence_select.main --data $DATA_DIR  --device 1 --feature_period $p --feature_used $feature --note all_but_discharge --task mortality --segment $s --model $m --split $split --filter physician)&
            done


            for p in all
            do
                (python -m models.sentence_select.main --data $DATA_DIR  --device 0 --feature_period retro --feature_used $feature --note $p --task readmission --segment $s --model $m --split $split --filter discharge)& #
            done

        done
        # sleep 30m
    done
done

#  Run important section of physician notes
(python -m models.sentence_select.main --data $DATA_DIR  --device 1 --feature_period 24 --feature_used $feature --note all_but_discharge --task mortality --segment section --model LR --split $split --filter physician)&

# # Run with all input with note filter
(python -m models.sentence_select.main --data $DATA_DIR  --device 1 --feature_period 24 --feature_used $feature --note all_but_discharge --task mortality --segment all --model LR --split $split --filter physician)&
(python -m models.sentence_select.main --data $DATA_DIR  --device 1 --feature_period retro --feature_used $feature --note all --task readmission --segment all --model LR --split $split --filter discharge)&

# # Run with all input without note filter
(python -m models.sentence_select.main --data $DATA_DIR  --device 1 --feature_period 24 --feature_used $feature --note all_but_discharge --task mortality --segment all --model LR --split $split )&
(python -m models.sentence_select.main --data $DATA_DIR  --device 1 --feature_period retro --feature_used $feature --note all --task readmission --segment all --model LR --split $split )&