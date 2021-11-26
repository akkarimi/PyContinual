#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1

for id in 0
do
    python run.py \
    --bert_model 'bert-base-uncased' \
    --note random$id \
    --ntasks 10 \
    --task femnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ucl_ncl \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --model_path "./models/fp32/til_classification/femnist/mlp_ucl_ncl_$id" \
    --save_model
done
#--nepochs 1
# lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100,