#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 5:00:00
#SBATCH -o dil_mnist_ncl_0-%j.out
#SBATCH --gres gpu:1


for id in 0 1 2 3 4
do
     python  run.py \
    --note random$id,online \
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_ncl \
    --train_batch_size 128 \
    --image_size 28 \
    --image_channel 1 \
    --nepochs 300 \
    --ent_id \
    --model_path "./models/fp32/til_classification/mnist/mlp_ncl_$id" \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/mnist/mlp_ncl_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done

# always set eval size to 1

#--nepochs 1
#    --train_data_size 500
