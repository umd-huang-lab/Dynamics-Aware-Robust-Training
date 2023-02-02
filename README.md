# Dynamics-Aware-Robust-Training
ICLR paper "Exploring and Exploiting Decision Boundary Dynamics for Adversarial Robustness" by Yuancheng Xu, Yanchao Sun, Micah Goldblum, Tom Goldstein and  Furong Huang

# DyART Documentation：

###### tags: `Codebase Doc`

## Enviroment
* Create a new enviroment using the yml file
> conda env create -f environment.yml
* Install auto attack
> pip install git+https://github.com/fra31/auto-attack


## Data Download
* Additional data for CIFAR-10

* TinyImagenet

## Running DyART

### CIFAR10 
Step 1.  Natural training for the burn-in period
> bash Cifar10_clean_training.sh

Step2. DyART traing
> Cifar10_DyART.sh

### CIFAR10 with additional data

### TinyImageNet

### Resuming 

### GroupNorm

### AutoAttack Evaluation
