import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from core.utils.train import SCHEDULERS

from core.utils.utils import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='Dynamics-Aware Robust Training and Baselines')
    parser.add_argument('--resume', type=str2bool, default=False, help='If yes, specify the resume file name; no need to specify any other arguments')
    parser.add_argument('--resume_fname', type=str,default=None)
    
    # DyART
    # h_prime 
    parser.add_argument('--h_prime_r0', type=str2float, default=16/255)
    parser.add_argument('--h_prime_alpha', type=float, default=0)
    parser.add_argument('--lam_robust', type=float, default=1000) 
    parser.add_argument('--temperature', type=float, default=5, help="Temperature of Cross-Entropy for soft bdr")
    parser.add_argument('--use_high_quality', type=str2bool, default=True, help='Check KKT condition for found FAB') 
    # FAB for bdr
    parser.add_argument('--restart_FAB', default=1, type=int) 
    parser.add_argument('--iter_FAB', default=20, type=int) 
    parser.add_argument('--eps_FAB', type=str2float, default=8/255) 
    
    # training 
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training.') # 1024
    parser.add_argument('--batch-size-validation', type=int, default=1024, help='Batch size for val and testing.') 
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimizer (SGD).') 
    parser.add_argument('--clip_grad', type=float, default=None, help='Gradient norm clipping.') # DyART needs gradient clipping
    parser.add_argument('--epochs', type=int, default=200, help='Number of all epochs.') 
    
    parser.add_argument('--DyART', type=str2bool, default=True, help='use DyART') 
    parser.add_argument("--pretrained", type=str, help="pretrained model path; None if not using any pretrained model", default=None) 
    parser.add_argument("--fname_extra", type=str, help="Extra info the file fname", default='')
   
    
    parser.add_argument("--clean_training", type=str2bool, default=False, help='Only clean training') 
    parser.add_argument("--save_intermediate_models", type=int, default=0) # epoch interval for saving; if 0, not save 
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional data.', 
                        default=None) 
    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of additional data to existing data.')
    parser.add_argument('--aux-ind-pth', type=str, default=None)
    
    ### others
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.') # cosinew
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    
    parser.add_argument('--model', choices=MODELS, default='wrn-28-10-swish', help='Model architecture to be used.')
    parser.add_argument('--beta', default=6.0, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.') # -1 then Madry's AT    
    parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.') 
    

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.') 
    parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')
    parser.add_argument('--mart', action='store_true', default=False, help='MART training.')
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')
    parser.add_argument('--data-dir', type=str, default='data/')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input before applying the model') 
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')   
    
    parser.add_argument('--GroupNorm', type=str2bool, default=False, help='If true: replacing BatchNorm with groupNorm') 
    
    return parser