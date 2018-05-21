#!/bin/bash

folders=(fiber_a,
         fiber_b,
         fiber_c,
         fiber_d,
         fiber_e)

for folder in "${folders[@]}"
do
    printf "Training is %s...\n" $folder
    python3 main.py \
        --csv_file $folder \
        --img_size 224 \
        --batch_size 64 \
        --optimizer Adam \
        --model resnet50 \
        --epochs 24
done
exit 0
