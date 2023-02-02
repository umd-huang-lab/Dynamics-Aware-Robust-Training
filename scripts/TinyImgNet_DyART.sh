#!/usr/bin/env bash

# TinyImgNet_clean_training.sh should be run first before running this script

cd ..
python train.py --data 'tiny-imagenet' --DyART true --batch-size-validation 1024 --batch-size 256 --data-dir 'data/' \
--lr 0.1  --scheduler 'cosinew' --epochs 100 \
--h_prime_r0 32/255 --h_prime_alpha 5 --lam_robust 500 --temperature 5 --iter_FAB 20 \
--model 'preact-resnet18' --GroupNorm False --clip_grad 1 \
--pretrained 'TinyImgNet_trained_models_Linf.results/Clean_preact-resnet18_lr_0.1_bs_256_42Epoch_BN.models/20.pt' --fname_extra 'fromClean20' \




## Group Normalization version
# cd ..
# python train.py --data 'tiny-imagenet' --DyART true --batch-size-validation 1024 --batch-size 256 --data-dir 'data/' \
# --lr 0.05  --scheduler 'cosinew' --epochs 100 \
# --h_prime_r0 20/255 --h_prime_alpha 3 --lam_robust 500 --temperature 5 --iter_FAB 20 \
# --model 'preact-resnet18' --GroupNorm True \
# --pretrained 'TinyImgNet_trained_models_Linf.results/Clean_preact-resnet18_lr_0.1_bs_256_42Epoch_GN.models/20.pt' --fname_extra 'fromClean20' \