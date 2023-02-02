"""
Evaluation with AutoAttack.

python eval-aa.py --fname_input xxx --eps_eval xxx --batch_size_for_eval xxx
"""

import json
import time
import argparse

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack

from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import seed


# Setup

def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')
    
    parser.add_argument('--norm_attack', type=str, default='Linf', choices = ['Linf', 'L2'])
    parser.add_argument('--eps_eval', type=float, default=8, help='Random seed.') # 8 for Linf, 0.5 for L2
    parser.add_argument('--fname_input', type=str, default='...')
    parser.add_argument('--batch_size_for_eval', type=int, default=1024) 
    
 
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='custom', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

parse = parser_eval()
args = parse.parse_args()

if args.norm_attack == 'Linf':
    eps_eval = args.eps_eval/255. # will use the eps specified by the parser_eval
else:
    eps_eval = args.eps_eval

# accessing and appending the args for training the model
with open(args.fname_input+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old) # new args = args from parser_eval and training args

DATA_DIR = args.data_dir + args.data
WEIGHTS = args.fname_input + '/val_best.pt'

log_path = args.fname_input + '/log-aa.log'
logger = Logger(log_path)
logger.log('\n\n')

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_for_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)

logger.log('evaluation data size:{}'.format(y_test.size(0)))
# Model

# +
model = create_model(args.model, args.normalize, info, device,GroupNorm=args.GroupNorm) ## dataParallel
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    logger.log('Using WA model.')
else:
    raise ValueError('Why not using WA model? Check again?')
    
try:
    model.load_state_dict(checkpoint['wa_model'])
except:
    model.module.load_state_dict(checkpoint['wa_model']) # when checkpt is not dataParallel
    
model.eval()
# -

# AA Evaluation

# +
seed(args.seed)
if args.norm_attack == 'Linf':
    assert args.attack in ['fgsm', 'linf-pgd', 'linf-df', 'linf-apgd']
elif args.norm_attack == 'L2':
    assert args.attack in ['fgm', 'l2-pgd', 'l2-df', 'l2-apgd']
else:
    raise ValueError('Invalid norm_attack for evaluation')

adversary = AutoAttack(model, norm=args.norm_attack, eps=eps_eval, log_path=log_path, version=args.version, seed=args.seed)
# -

logger.log('{} AA evaluation on:\n{}\n'.format(args.norm_attack, WEIGHTS))
try:
    logger.log('epoch {} with val_best {}'.format(checkpoint['epoch'],checkpoint['val_best']))
except:
    logger.log('epoch {} with test_best {}'.format(checkpoint['epoch'],checkpoint['test_best']))
del checkpoint

logger.log('eps:{:.4f} batch size:{}\n'.format(eps_eval,BATCH_SIZE_VALIDATION))

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')
