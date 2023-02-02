#!/usr/bin/env bash

# Cifar10_clean_training.sh should be run first before running this script
# cifar10_ddpm.npz needs to be first downloaded in the main directory

cd ../
python train.py --data 'cifar10s' --DyART true --batch-size-validation 1024 --batch-size 512 \
--lr 0.1  --scheduler 'cosinew' --epochs 800 \
--aux-data-filename 'cifar10_ddpm.npz' --unsup-fraction 0.7 \
--h_prime_r0 16/255 --h_prime_alpha 3 --lam_robust 800 --temperature 5 --iter_FAB 20 \
--model 'wrn-28-10-swish' --GroupNorm False --clip_grad 0.1 \
--pretrained 'Cifar10_trained_models_Linf.results/Clean_wrn-28-10-swish_lr_0.1_bs_256_22Epoch_BN.models/10.pt' --fname_extra 'fromClean10_AuxData' \

