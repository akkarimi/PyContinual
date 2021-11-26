#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 15:00:00
#SBATCH -o til_asc_kim_acl-%j.out
#SBATCH --gres gpu:1


for id in  4
do
     python run.py \
    --note random$id \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_acl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/femnist/mlp_acl_$id" \
    --save_model
done
