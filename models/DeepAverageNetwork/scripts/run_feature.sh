#!/bin/bash
set -x

for i in 24 48 retro
do
for n in physician physician_nursing all_but_discharge
do
   (python -m main -device 1 -name $n -period $i -feature -task mortality -learning_rate 0.0001 -epoch 10) 
done
done
#for i in physician discharge
for i in all discharge
do
    (python -m main -device 1 -name $i -period retro -feature -task readmission -learning_rate 0.0001 -epoch 10)
done
