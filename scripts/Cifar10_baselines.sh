#!/usr/bin/env bash

####### Baselines #######
### Trades
# cd ..
# python train.py --data 'cifar10s' --DyART False --batch-size-validation 1024 --batch-size 256 \
# --lr 0.1  --scheduler 'cosinew' --epochs 200 \
# --beta 6.0 \
# --model 'wrn-28-10-swish' --GroupNorm False \

### AT
# cd ..
# python train.py --data 'cifar10s' --DyART False --batch-size-validation 1024 --batch-size 256 \
# --lr 0.1  --scheduler 'cosinew' --epochs 200 \
# --beta -1 \
# --model 'wrn-28-10-swish' --GroupNorm False \

### MART
# cd ..
# python train.py --data 'cifar10s' --DyART False --batch-size-validation 1024 --batch-size 256 \
# --lr 0.1  --scheduler 'cosinew' --epochs 200 \
# --beta 6.0 --mart \
# --model 'wrn-28-10-swish' --GroupNorm False \


## to run the Group Normalization version, just add the flag: --GroupNorm true
