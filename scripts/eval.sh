#!/usr/bin/env bash

function runexp {
fname=${1}
eps=${2}
norm=${3}

echo "norm = $norm, eps = $eps, fname = $fname"

python ../eval_aa.py --fname_input $fname --eps_eval $eps --norm_attack $norm --batch_size_for_eval 1500
}


############# run ############# 

### Linf

norm='Linf'
# eps_list=(8)
eps_list=(2 4 8 12 16)


declare -a fname_list=(
'../Cifar10_trained_models_Linf.results/h_prime_exp_alpha_3.0_r0_16_funcTemp_5.0_lr_0.1_iterFAB_20_lamRobust_1000.0_bs_256_clipGrad_0.1_fromClean10_200Epoch_BN.models'
)


for fname in "${fname_list[@]}"
do
       for eps in ${eps_list[@]}
       do
              runexp $fname $eps $norm
       done

done





