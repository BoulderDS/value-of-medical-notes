#!/bin/bash

export DATA_DIR=/data/test_mimic_output/
export VOCAB_DIR=$DATA_DIR/Deep-Average-Network/
mkdir -p $VOCAB_DIR

echo "build vocab for readmission prediction"
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period retro --note discharge --task readmission > /dev/null 2>&1 &
python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR  --period retro --note all --task readmission & # > /dev/null 2>&1 &

echo "build vocab for mortality prediction"
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period 24 --note physician --task mortality > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period 48 --note physician --task mortality > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period retro --note physician --task mortality > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period 24 --note physician_nursing --task mortality > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period 48 --note physician_nursing --task mortality > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period retro --note physician_nursing --task mortality > /dev/null 2>&1 &
python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period 24 --note all_but_discharge --task mortality  & # > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period 48 --note all_but_discharge --task mortality > /dev/null 2>&1 &
# python -m build_vocab --vocab_dir $VOCAB_DIR --data_dir $DATA_DIR --period retro --note all_but_discharge --task mortality > /dev/null 2>&1 &
