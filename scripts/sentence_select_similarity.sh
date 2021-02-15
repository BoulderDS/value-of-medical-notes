#!/bin/bash
set -x
feature="notes"
split="test"
model="LR" # or "DAN" 
export DATA_DIR=/data/test_mimic_output/
for t in  similaritymix_0.5 similarity_far similarity_near  # sentsimilarity_near  sentsimilarity_far  sentsimilaritymix_0.5  #  similarity_near  similarity_far similaritymix_0.5 fracmedicalsent_sim_norm # fracmedicalsent_sim countmedicalsent_sim longest_sim  # similarity_near  similarity_far similaritymix_0.5 # similaritymix_0.2 similaritymix_0.4 similaritymix_0.6 similaritymix_0.8  #  fracmedicalsent_sim_norm fracmedicalsent_sim countmedicalsent_sim longest_sim  section_sim all_sim # 
do
    for m in $model  #  
    do
        for p in $t\_max_new_$split # $t\_avg_new_$split # $t\_avg_new_norm_$split    $t\_max_new_norm_$split  # $t\_$split # 
        do
            for s in $p\_token\_percent\_1 $p\_token\_percent\_5   $p\_token\_percent\_10 $p\_token\_percent\_20  $p\_token\_percent\_50 $p\_token\_percent\_15 $p\_token\_percent\_25 $p\_token\_percent\_30 $p\_token\_percent\_35 $p\_token\_percent\_40 $p\_token\_percent\_45  # $p\_token\_100  $p\_token\_200 $p\_token\_400 # $p\_1  $p\_5  $p\_10 $p\_20 $p\_50  $p\_token\_800  $p\_token\_1000 $p\_token\_1600 $p\_token\_2000 
            do
                for p in 24 # 48 retro
                do
                    (python -m models.sentence_select.main --data $DATA_DIR --device 0 --feature_period $p --feature_used $feature --note all_but_discharge --task mortality --segment $s --model $m --filter physician --split $split)&
                done


                for p in all
                do
                    (python -m models.sentence_select.main --data $DATA_DIR --device 1 --feature_period retro --feature_used $feature --note $p --task readmission --segment $s --model $m --filter discharge --split $split)& #--filter discharge
                done
                sleep 5s
            done
        done
    done
done