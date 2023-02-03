#!/usr/bin/env bash

cd ../
python train.py --resume True \
--resume_fname '**provide the file directory here; All the argument parameters used in previous unfinished experiments are automatically loaded**'

# an example
# cd ../
# python train.py --resume True \
# --resume_fname 'Cifar10_trained_models_Linf.results/h_prime_exp_alpha_3.0_r0_16_funcTemp_5.0_lr_0.1_iterFAB_20_lamRobust_1000.0_bs_256_clipGrad_0.1_fromClean10_200Epoch_BN.models'

