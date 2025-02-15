#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 20:00:00
#SBATCH -o til_dsc_kim_gem_2-%j.out
#SBATCH --gres gpu:1

#TODO: GEM baseline, could be time-consuming, consider A-GEM
#TODO： need to change...


for id in 2
do
    python  run.py \
    --bert_model 'bert-base-uncased' \
    --experiment bert \
    --approach bert_kim_a-gem_ncl \
    --note random$id,200 \
    --ntasks 10 \
    --task dsc \
    --idrandom $id \
    --output_dir './OutputBert' \
    --scenario til_classification \
    --train_data_size 200 \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --nepochs 100 \
    --buffer_size 128 \
    --buffer_percent 0.02 \
    --gamma 0.5
done
#--nepochs 1
