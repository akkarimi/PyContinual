#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --experiment w2v \
    --note random$id \
    --ntasks 19 \
    --idrandom $id \
    --approach w2v_kim_pcnn_one
done
