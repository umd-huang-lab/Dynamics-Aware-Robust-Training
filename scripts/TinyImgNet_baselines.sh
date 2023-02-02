#!/usr/bin/env bash

########### Baselines ##########
### AT
# cd ..
# python train.py --GroupNorm False --data 'tiny-imagenet' --model 'preact-resnet18' \
# --DyART False --epochs 100 \
# --lr 0.1  --scheduler 'cosinew' --batch-size 256 --batch-size-validation 1024 \
# --beta -1 \
# --data-dir 'data/' \

### Trades
# cd ..
# python train.py --GroupNorm False --data 'tiny-imagenet' --model 'preact-resnet18' \
# --DyART False --epochs 100 \
# --lr 0.1  --scheduler 'cosinew' --batch-size 256 --batch-size-validation 1024 \
# --beta 6.0 \
# --data-dir 'data/' \

### MART
# cd ..
# python train.py --GroupNorm False --data 'tiny-imagenet' --model 'preact-resnet18' \
# --DyART False --epochs 100 \
# --lr 0.1  --scheduler 'cosinew' --batch-size 256 --batch-size-validation 1024 \
# --beta 6.0 --mart \
# --data-dir 'data/' \


## to run the Group Normalization version, just add the flag: --GroupNorm true



