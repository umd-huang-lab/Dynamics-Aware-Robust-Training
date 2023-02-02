#!/usr/bin/env bash

# Cifar10_clean_training.sh should be run first before running this script

cd ../
python train.py --data 'cifar10s' --DyART true --batch-size-validation 1024 --batch-size 256 \
--lr 0.1  --scheduler 'cosinew' --epochs 200 \
--h_prime_r0 16/255 --h_prime_alpha 3 --lam_robust 1000 --temperature 5 --iter_FAB 20 \
--model 'wrn-28-10-swish' --GroupNorm False --clip_grad 0.1 \
--pretrained 'Cifar10_trained_models_Linf.results/Clean_wrn-28-10-swish_lr_0.1_bs_256_22Epoch_BN.models/10.pt' --fname_extra 'fromClean10' \

### Group Normalization version
# cd ../
# python train.py --data 'cifar10s' --DyART true --batch-size-validation 1024 --batch-size 256 \
# --lr 0.1  --scheduler 'step' --epochs 100 \
# --h_prime_r0 16/255 --h_prime_alpha 8 --lam_robust 400 --temperature 5 --iter_FAB 20 \
# --model 'wrn-28-10-swish' --GroupNorm true \
# --pretrained 'Cifar10_trained_models_Linf.results/Clean_wrn-28-10-swish_lr_0.1_bs_256_22Epoch_GN.models/10.pt' --fname_extra 'fromClean10' \

