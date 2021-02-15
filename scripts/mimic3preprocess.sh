#!/usr/bin/env bash

export NUM_WORKER=64 # number of threads for multithreading
export DATA_DIR=/data/joe/physician_notes/mimic-data/ #path to your mimic csv files 
export OUTPUT_DIR=/data/test_mimic_output/

mkdir -p $OUTPUT_DIR

set -x
echo "Extract Subjects"
python -m  mimic3preprocess.scripts.extract_subjects $DATA_DIR $OUTPUT_DIR --n_workers $NUM_WORKER
echo "Train Test Split"
python -m mimic3preprocess.scripts.split_train_and_test $OUTPUT_DIR
echo "Extract Episodes from Subjects wit CGID"
python -m mimic3preprocess.scripts.extract_episodes_from_subjects_multiprocessing $OUTPUT_DIR $NUM_WORKER


echo "Create features in 24 hours"
python -m mimic3preprocess.scripts.feature_extraction_multiprocessing --num_worker $NUM_WORKER --period_length 24 --output_dir $OUTPUT_DIR/features_24/
# echo "Create features in 48 hours"
# python -m mimic3preprocess.scripts.feature_extraction_multiprocessing --num_worker $NUM_WORKER --period_length 48 --output_dir $OUTPUT_DIR/features_48/
echo "Create features in all admission"
python -m mimic3preprocess.scripts.feature_extraction_multiprocessing --num_worker $NUM_WORKER --period_length -1 --output_dir $OUTPUT_DIR/features_retro/

echo "Merge features in 24 hours"
python -m mimic3preprocess.scripts.merge_features  ./mimic3preprocess/resources/ $OUTPUT_DIR/features_24/
# echo "Merge features in 48 hours"
# python -m mimic3preprocess.scripts.merge_features $DATA_DIR/in_hospital_retro/ $OUTPUT_DIR/features_48/
echo "Merge features in all admission"
python -m mimic3preprocess.scripts.merge_features  ./mimic3preprocess/resources/ $OUTPUT_DIR/features_retro/

echo "Create timeseries features in 24 hours"
python -m mimic3preprocess.scripts.timeseries_feature_extraction_multiprocessing --num_worker $NUM_WORKER --period_length 24 --output_dir $OUTPUT_DIR/timeseries_features_24/
# echo "Create timeseries features in 48 hours"
# python -m mimic3preprocess.scripts.timeseries_feature_extraction_multiprocessing --num_worker $NUM_WORKER --period_length 48 --output_dir $OUTPUT_DIR/timeseries_features_48/
echo "Create timeseries features in all admission"
python -m mimic3preprocess.scripts.timeseries_feature_extraction_multiprocessing --num_worker $NUM_WORKER --period_length -1 --output_dir $OUTPUT_DIR/timeseries_features_retro/

######################
# Start process mortality
######################

echo "Create In-Hospital Mortality for 24 hours"
(python -m mimic3preprocess.scripts.create_in_hospital_mortality $OUTPUT_DIR $OUTPUT_DIR/mortality/ 24)
#echo "Create In-Hospital Mortality for 48 hours"
#(python -m mimic3preprocess.scripts.create_in_hospital_mortality $OUTPUT_DIR $OUTPUT_DIR/mortality/ 48)
#echo "Create In-Hospital Mortality for retro"
#(python -m mimic3preprocess.scripts.create_in_hospital_mortality $OUTPUT_DIR $OUTPUT_DIR/mortality/ -1)

echo "Train valid split mortality prediction task"
python -m mimic3preprocess.scripts.get_validation --path $OUTPUT_DIR --task mortality --period 24
#python -m mimic3preprocess.scripts.get_validation --path $OUTPUT_DIR --task mortality --period 48
#python -m mimic3preprocess.scripts.get_validation --path $OUTPUT_DIR --task mortality --period retro

echo "Create Notes for mortality prediction"
for n in all_but_discharge #physician physician_nursing all_but_discharge all
do
(python -m mimic3preprocess.scripts.create_in_hospital_mortality_note $OUTPUT_DIR $OUTPUT_DIR/mortality/ $n 24)&
#(python -m mimic3preprocess.scripts.create_in_hospital_mortality_note $OUTPUT_DIR $OUTPUT_DIR/mortality/ $n 48)&
#(python -m mimic3preprocess.scripts.create_in_hospital_mortality_note $OUTPUT_DIR $OUTPUT_DIR/mortality/ $n -1)&
#done


echo "Create list of admission with at least one note (except discharge summary) for mortality given time span"
for p in 24  #48 retro
do
    for n in all_but_discharge # physician  physician_nursing
    do
    python -m mimic3preprocess.scripts.get_data_with_notes --path $OUTPUT_DIR --task mortality --period $p --note $n
    done
done

######################
# Start process readmission
######################
echo "Create Readmission task"
python -m mimic3preprocess.scripts.create_readmission $OUTPUT_DIR $OUTPUT_DIR/readmission/

echo "Create list of admission with at least one note for readmission"
for n in  all #discharge
do
(python -m mimic3preprocess.scripts.create_in_hospital_mortality_note $OUTPUT_DIR $OUTPUT_DIR/readmission/ $n -1)&
done

echo "Train/Val/Test split on readmission task"
python -m mimic3preprocess.scripts.get_validation --path $OUTPUT_DIR --task readmission --period retro

echo "Create data with note"
for p in retro
do
    for n in all #all_but_discharge # physician  physician_nursing
    do
    python -m mimic3preprocess.scripts.get_data_with_notes --path $OUTPUT_DIR --task readmission --period $p --note $n
    done
done


# echo "Create Only Discharge Notes"
# python -m mimic3preprocess.scripts.create_in_hospital_mortality_discharge_note /data/joe/physician_notes/mimic-data/preprocessed/ /data/joe/physician_notes/mimic-data/in_hospital_retro_discharge_note/ -1


########
# GRU-D
########
#echo "Compute X_mean features in 24 hours for GRU-D"
#python -m mimic3preprocess.scripts.calculate_mean_x /data/joe/physician_notes/mimic-data/mortality/ /data/joe/physician_notes/mimic-data/preprocessed/grud_features_24/ physician 24
#echo "Compute X_mean features in 48 hours for GRU-D"
#python -m mimic3preprocess.scripts.calculate_mean_x /data/joe/physician_notes/mimic-data/mortality/ /data/joe/physician_notes/mimic-data/preprocessed/grud_features_48/ physician 48
#echo "Compute X_mean features in retro hours for GRU-D"
#python -m mimic3preprocess.scripts.calculate_mean_x /data/joe/physician_notes/mimic-data/mortality/ /data/joe/physician_notes/mimic-data/preprocessed/grud_features_retro/ physician retro

#echo "Compute X_mean features in 24 hours for GRU-D"
#python -m mimic3preprocess.scripts.calculate_all_mean_x /data/joe/physician_notes/mimic-data/mortality/ /data/joe/physician_notes/mimic-data/preprocessed/features_24.pkl physician 24
#echo "Compute X_mean features in 48 hours for GRU-D"
#python -m mimic3preprocess.scripts.calculate_all_mean_x /data/joe/physician_notes/mimic-data/mortality/ /data/joe/physician_notes/mimic-data/preprocessed/features_48.pkl physician 48
#echo "Compute X_mean features in retro hours for GRU-D"
#python -m mimic3preprocess.scripts.calculate_all_mean_x /data/joe/physician_notes/mimic-data/mortality/ /data/joe/physician_notes/mimic-data/preprocessed/features_retro.pkl physician retro

#echo "Create Only Physician Notes"
#python -m mimic3preprocess.scripts.create_in_hospital_mortality_physician_note /data/joe/physician_notes/mimic-data/preprocessed/ /data/joe/physician_notes/mimic-data/in_hospital_retro_physician_note/ -1
#echo "Create Only Physician Notes"
#python -m mimic3preprocess.scripts.create_in_hospital_mortality_physician_note /data/joe/physician_notes/mimic-data/preprocessed/ /data/joe/physician_notes/mimic-data/in_hospital_24_physician_note/ 24
#echo "Create Only Physician Notes"
#python -m mimic3preprocess.scripts.create_in_hospital_mortality_physician_note /data/joe/physician_notes/mimic-data/preprocessed/ /data/joe/physician_notes/mimic-data/in_hospital_48_physician_note/ 48

