#!/bin/bash

if [ ! -d "OutputBert" ]; then
  mkdir OutputBert
fi

for id in 0 1 3
do
    CUDA_VISIBLE_DEVICES=0 python  run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id,epoch24\
    --ntasks 2 \
    --task asc \
    --scenario til_classification \
    --idrandom $id \
    --output_dir './OutputBert' \
    --approach bert_adapter_capsule_ncl \
    --experiment bert_adapter_capsule \
    --eval_batch_size 32 \
    --train_batch_size 16 \
    --num_train_epochs 10 \
    --apply_bert_output \
    --apply_bert_attention_output \
    --build_adapter_capsule  \
    --apply_two_layer_shared \
    --xusemeval_num_train_epochs 10 \

done
