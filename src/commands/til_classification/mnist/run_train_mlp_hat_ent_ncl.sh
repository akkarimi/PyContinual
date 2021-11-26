#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 1:00:00
#SBATCH -o dil_asc_adapter_mask_mixup_1-%j.out
#SBATCH --gres gpu:1


for id in 3 4
do
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --note random$id,online\
    --ntasks 10 \
    --task mnist \
    --scenario til_classification \
    --idrandom $id \
    --approach mlp_hat_ncl \
    --image_size 28 \
    --image_channel 1 \
    --nepochs 1000 \
    --ent_id \
    --resume_model \
    --eval_only \
    --resume_from_file "./models/fp32/til_classification/mnist/mlp_hat_$id" \
    --resume_from_aux_file "./models/fp32/til_classification/mnist/mlp_hat_$id" \
    --resume_from_task 9 \
    --eval_batch_size 1
done
#--nepochs 1
#    --train_batch_size 128 \
#-m torch.distributed.launch --nproc_per_node=2
#    --distributed \
#    --multi_gpu \
#    --ngpus 2

#    --lr 0.05 \
#    --lr_min 1e-4 \
#    --lr_factor 3 \
#    --lr_patience 5 \
#    --clipgrad 10000

#semantic cap size 1000, 500, 2048

#    --model_path "./models/fp32/dil_classification/celeba/cnn_hat_amix_ent_$id" \
#    --save_model