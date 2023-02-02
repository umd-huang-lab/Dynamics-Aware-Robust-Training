#!/usr/bin/env bash

# Only need to run 20 epochs for DyART's burn-in period. Here 42 epochs is used since 'step' learning rate schedule is used, which decays the learning rate at epoch 21. You can stop the program once 20 epochs has finished.

cd ../
python train.py --data 'tiny-imagenet' --batch-size-validation 1024 --batch-size 256 --data-dir 'data/' \
--clean_training true --epochs 42 --save_intermediate_models 10 \
--lr 0.1  --scheduler 'step' \
--model 'preact-resnet18' --GroupNorm False \



### Group Normalization version
# cd ../
# python train.py --data 'tiny-imagenet' --batch-size-validation 1024 --batch-size 256 --data-dir 'data/' \
# --clean_training true --epochs 42 --save_intermediate_models 10 \
# --lr 0.1  --scheduler 'step' \
# --model 'preact-resnet18' --GroupNorm true \
