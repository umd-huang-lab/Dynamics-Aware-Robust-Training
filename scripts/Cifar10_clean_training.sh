#!/usr/bin/env bash

# Only need to run 10 epochs for DyART's burn-in period. Here 22 epochs is used since 'step' learning rate schedule is used, which decays the learning rate at epoch 11. You can stop the program once 10 epochs has finished.

cd ../
python train.py --data 'cifar10s' --clean_training true --batch-size-validation 1024 --batch-size 256 \
--epochs 22 --save_intermediate_models 10 \
--lr 0.1  --scheduler 'step' \
--model 'wrn-28-10-swish' --GroupNorm False \


### Group Normalization version
# cd ../
# python train.py --data 'cifar10s' --clean_training true --batch-size-validation 1024 --batch-size 256 \
# --epochs 22 --save_intermediate_models 10 \
# --lr 0.1  --scheduler 'step' \
# --model 'wrn-28-10-swish' --GroupNorm true \

