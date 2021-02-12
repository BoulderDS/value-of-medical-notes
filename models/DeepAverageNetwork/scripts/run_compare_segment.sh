#!/bin/bash

# for w in 1 5 10
# do
# (python compare_notes.py -task mortality -name all_but_discharge -period 24 -text  -device 1 -compare_note medical_$w)&
# (python compare_notes.py -task mortality -name all_but_discharge -period 48 -text  -device 0 medical_$w)&
# (python compare_notes.py -task mortality -name all_but_discharge -period retro -text  -device 0 medical_$w)&

# (python compare_notes.py -task readmission -name all -period retro -text  -device 1  medical_$w)&

# (python compare_notes.py -task mortality -name all_but_discharge -period 24 -text -feature -device 1  medical_$w)&
# (python compare_notes.py -task mortality -name all_but_discharge -period 48 -text -feature -device 0  medical_$w)&
# (python compare_notes.py -task mortality -name all_but_discharge -period retro -text -feature -device 0  medical_$w)&

# (python compare_notes.py -task readmission -name all -period retro -text -feature -device 1 medical_$w)
# done
for i in  5 6 7 8 9 #0 1 2 3  4
do
    for s in sample_$i\_token_50 sample_$i\_token_250 sample_$i\_token_500 sample_$i\_token_1000 
    do
        (python compare_notes.py -task mortality -name all_but_discharge -period 24 -text  -device 1 -segment $s)&
        (python compare_notes.py -task mortality -name all_but_discharge -period 48 -text  -device 0 -segment $s)&
        (python compare_notes.py -task mortality -name all_but_discharge -period retro -text  -device 0 -segment $s)&

        (python compare_notes.py -task readmission -name all -period retro -text  -device 1  -segment $s)&

        # (python compare_notes.py -task mortality -name all_but_discharge -period 24 -text -feature -device 1  -segment $s)&
        # (python compare_notes.py -task mortality -name all_but_discharge -period 48 -text -feature -device 0  -segment $s)&
        # (python compare_notes.py -task mortality -name all_but_discharge -period retro -text -feature -device 0  -segment $s)&

        # (python compare_notes.py -task readmission -name all -period retro -text -feature -device 1 -segment $s)&
        sleep 3m
    done
done