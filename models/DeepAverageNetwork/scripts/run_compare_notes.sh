#!/bin/bash

(python compare_notes.py -task mortality -name all_but_discharge -period 24 -text  -device 1)&
(python compare_notes.py -task mortality -name all_but_discharge -period 48 -text  -device 0)&
(python compare_notes.py -task mortality -name all_but_discharge -period retro -text  -device 0)&

(python compare_notes.py -task readmission -name all -period retro -text  -device 1)&

(python compare_notes.py -task mortality -name all_but_discharge -period 24 -text -feature -device 1)&
(python compare_notes.py -task mortality -name all_but_discharge -period 48 -text -feature -device 0)&
(python compare_notes.py -task mortality -name all_but_discharge -period retro -text -feature -device 0)&

(python compare_notes.py -task readmission -name all -period retro -text -feature -device 1)&
