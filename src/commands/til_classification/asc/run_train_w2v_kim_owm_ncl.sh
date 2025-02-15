#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 4
do
    CUDA_VISIBLE_DEVICES=0 python  main.py \
    --bert_model 'bert-base-uncased' \
    --experiment w2v \
    --note random$id \
    --ntasks 19 \
    --tasknum 19 \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach w2v_kim_owm_ncl \
    --optimizer SGD
done
#--nepochs 1