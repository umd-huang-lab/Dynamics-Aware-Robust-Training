import json
import time
import argparse

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data
from core.data import SEMISUP_DATASETS

from core.utils import format_time
from core.utils import Logger
from core.utils import seed

from core.utils_gowal21 import WATrainer

from parser import parser_train


# Setup

def get_fname(args):
    assert args.DyART == True, 'get_fname(): DyART is not true'
    arch_name = ''
    
    if args.norm_type == 'Linf':
        h_prime_detail = 'h_prime' + '_exp_alpha_' + str(args.h_prime_alpha) + \
        '_r0_' + str(int(args.h_prime_r0*255))
    else:
        h_prime_detail = 'h_prime' + '_exp_alpha_' + str(args.h_prime_alpha) + \
        '_r0_' + str(args.h_prime_r0)
        
    if not args.resume and not args.clean_training:
        print('scheme for h_prime is exponential decay (only include R<r0)')
        
    func_detail = '_funcTemp_' + str(args.temperature) 

    train_detail = arch_name + '_lr_' + str(args.lr) + \
        '_iterFAB_' + str(args.iter_FAB) + \
        '_lamRobust_' + '{}'.format(args.lam_robust) + \
        '_bs_' + str(args.batch_size) + '_clipGrad_' + str(args.clip_grad)
        
    normalization_layer_name = '_GN' if args.GroupNorm else '_BN'
    
    fname_extra = args.fname_extra + '_{}Epoch'.format(args.epochs) + normalization_layer_name

    fname = h_prime_detail + func_detail + train_detail + \
             '_' + fname_extra + '.models'
    
    if args.clean_training:
        fname = 'Clean_' + args.model + '_lr_' + str(args.lr) + \
                '_bs_' + str(args.batch_size) + fname_extra + '.models'

    if args.data == 'tiny-imagenet':
        fname = os.path.join('TinyImgNet_trained_models_{}.results'.format(args.norm_type),fname)
    elif args.data == 'cifar10s':
        fname = os.path.join('Cifar10_trained_models_{}.results'.format(args.norm_type),fname)
    elif args.data == 'cifar100s':
        fname = os.path.join('Cifar100_trained_models_{}.results'.format(args.norm_type),fname)
    else:
        raise ValueError('Only support cifar10s, cifar100s and tiny-imagenet')
    
    
    return fname


# +
parse = parser_train()
parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
args = parse.parse_args()

assert args.data in SEMISUP_DATASETS or args.data == 'tiny-imagenet', f'Only data in {SEMISUP_DATASETS} is supported!' 

if args.attack in ['fgsm', 'linf-pgd', 'linf-df', 'linf-apgd']:
    args.norm_type = 'Linf'
elif args.attack in ['fgm', 'l2-pgd', 'l2-df', 'l2-apgd']:
    args.norm_type = 'L2'
print('{}\n'.format(args.norm_type))

if args.DyART:
    args.fname = get_fname(args)
else:
    fname_extra = args.fname_extra + '_{}Epoch'.format(args.epochs)
    normalization_layer_name = '_GN' if args.GroupNorm else '_BN'
    fname_extra += normalization_layer_name

    
if not args.resume:
    if not args.clean_training:
        if args.DyART:
            print('DyART Training')
        elif args.beta > 0 and args.mart:
            print('MART Training')
            fname_extra = args.norm_type + '_' + 'MART' + fname_extra
        elif args.beta > 0:
            print('TRADES Training')
            fname_extra = args.norm_type + '_' + 'Trades' + fname_extra
        else:
            # beta < = 0
            print('AT Training')
            fname_extra = args.norm_type + '_' + 'AT' + fname_extra
    else:
        print('Clean Training')
    
    if not args.DyART:
        if args.data == 'tiny-imagenet':
            fname_extra = 'TinyImgNet_' + fname_extra
        elif args.data == 'cifar100s':
            fname_extra = 'CIFAR100_' + fname_extra
        else:
            fname_extra = 'CIFAR10_' + fname_extra
        args.fname = os.path.join('trained_model_baseline',fname_extra)

if os.path.exists(args.fname) and not args.resume:
    print('\n\n\n\nThe file name already exists. Maybe check your hyperparameters or delete the file? {}\n\n\n'.format(args.fname))
#     raise ValueError('The file name already exists. Maybe check your hyperparameters or delete the file? {}'.format(args.fname))

if not os.path.exists(args.fname) and not args.resume:
    os.makedirs(args.fname)

if args.resume:
    # load all training parameter from file; later, use args.fname (args.resume_fname is meaningless after loading)
    args = torch.load(os.path.join(args.resume_fname, 'ResumeParameter.pth'))
    args.resume = True
    args.resume_fname = args.fname
    print(args)
else:
    with open(os.path.join(args.fname, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
        
    torch.save(args, os.path.join(args.fname, 'ResumeParameter.pth') )

DATA_DIR = os.path.join(args.data_dir, args.data)
logger = Logger(os.path.join(args.fname, 'log-train.log'))
# -


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
EPOCHS = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.debug:
    EPOCHS = 1
# To speed up training
torch.backends.cudnn.benchmark = True

seed(args.seed)
train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, aux_ind_pth = args.aux_ind_pth, \
    unsup_fraction=args.unsup_fraction, validation=True
)
logger.log('\n\n{}'.format(args))
SEMISUP_DATASETS = ['cifar10s', 'cifar100s']
dataset = os.path.basename(os.path.normpath(DATA_DIR))
if dataset in SEMISUP_DATASETS:
    logger.log('Size of aux data: {}\n'.format(len(train_dataset.unsup_indices)))
del train_dataset, test_dataset, val_dataset

seed(args.seed)
if args.tau:
    print ('Using WA.')
    trainer = WATrainer(info, args)
else:
    raise ValueError('Should use WA!')
    trainer = Trainer(info, args)

if EPOCHS > 0:
    metrics = pd.DataFrame()
    trainer.init_optimizer(args.epochs)
        
    # best adv acc
    val_best = 0 
    test_best = 0
    
    start_epoch = 0


# +
if args.resume:
    checkpoint = torch.load(os.path.join(trainer.params.fname, 'latest_checkpoint.pt'))

    trainer.model.module.load_state_dict(checkpoint['unaveraged_model'])
    trainer.wa_model.module.load_state_dict(checkpoint['wa_model'])
    
    trainer.optimizer.load_state_dict(checkpoint['optimizer']) 
    trainer.scheduler.load_state_dict(checkpoint['scheduler']) 
    
    val_best = checkpoint['val_best']
    test_best = checkpoint['test_best']
    
    start_epoch = trainer.scheduler.last_epoch
    
    if trainer.params.scheduler not in ['cyclic']:
        assert checkpoint['epoch'] + 1 == start_epoch, 'Resuming: start_epoch is wrong! checkpoint_epoch + 1:{} start_epoch:{}'.format(checkpoint['epoch'] + 1, start_epoch)
    else:
        start_epoch = checkpoint['epoch'] + 1
    
    logger.log('Resuming from epoch {} with current val_best:{:.4f} and test_best:{:.4f} from {}'.format(\
                 start_epoch,val_best,test_best,args.fname))
elif args.pretrained is not None:
    # Not resume but use pretrained model
#     logger.log('loading (unaveraged) pretrained model from {}'.format(args.pretrained))
#     trainer.model.module.load_state_dict(torch.load(args.pretrained)['unaveraged_model'])
#     trainer.wa_model.module.load_state_dict(torch.load(args.pretrained)['unaveraged_model'])
    
    logger.log('loading (Weighted-averaged) pretrained model from {}'.format(args.pretrained))
    trainer.model.module.load_state_dict(torch.load(args.pretrained)['wa_model'])
    trainer.wa_model.module.load_state_dict(torch.load(args.pretrained)['wa_model'])

    
if not args.clean_training:
    logger.log('Val Clean: {:.4f}%\tRobust: {:.4f}%'.format(trainer.eval(val_dataloader)*100,trainer.eval(val_dataloader, adversarial=True)*100))
else:
    logger.log('Val Clean: {:.4f}%, Robust is expected to be zero since we are doing clean training'.format(trainer.eval(val_dataloader)*100))
    
# -

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    last_lr = trainer.scheduler.get_last_lr()[0]
    print('last_lr is :{}'.format(last_lr))
    print(trainer.params.fname)
    
    if not args.clean_training:
        train_stat = trainer.train(train_dataloader, epoch=epoch, adversarial=True) # 'loss', 'clean_acc' and 'adversarial_acc'
        train_loss, train_clean_acc, train_adv_acc = train_stat['loss'], train_stat['clean_acc'], train_stat['adversarial_acc']
        logger.log('Training Loss: {:.4f}.\tLR: {:.4f}'.format(train_loss, last_lr))

        test_clean_acc = trainer.eval(test_dataloader)
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)

        val_clean_acc = trainer.eval(val_dataloader)
        val_adv_acc = trainer.eval(val_dataloader, adversarial=True)

        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tVal: {:.2f}%.\tTest: {:.2f}%.'.format(\
                        train_clean_acc*100, val_clean_acc*100, test_clean_acc*100))
        logger.log('Robust Accuracy-\tTrain: {:.2f}%.\tVal: {:.2f}%.\tTest: {:.2f}%.'.format(\
                        train_adv_acc*100, val_adv_acc*100, test_adv_acc*100))

        epoch_metrics = {'train_loss':train_loss, 'train_clean_acc':train_clean_acc, 'train_adv_acc':train_adv_acc,\
                        'test_clean_acc':test_clean_acc, 'test_adv_acc':test_adv_acc,\
                        'val_clean_acc':val_clean_acc, 'val_adv_acc':val_adv_acc,\
                        'epoch': epoch, 'lr': last_lr} 
        
        if val_adv_acc > val_best:
            val_best = val_adv_acc
            print('saving val_best: {:.2f}%'.format(val_best * 100))
            torch.save({'unaveraged_model': trainer.model.module.state_dict(),
                   'wa_model': trainer.wa_model.module.state_dict(),
                   'val_best':val_best, 'epoch':epoch}, \
                   os.path.join(trainer.params.fname, 'val_best.pt'))

        if test_adv_acc > test_best:
            test_best = test_adv_acc
            print('saving test_best: {:.2f}%'.format(test_best * 100))
            torch.save({'unaveraged_model': trainer.model.module.state_dict(),
                   'wa_model': trainer.wa_model.module.state_dict(),
                   'test_best':test_best, 'epoch':epoch}, \
                   os.path.join(trainer.params.fname, 'test_best.pt'))
        
    else:
        # clean training
        logger.log('Clean training')
        train_stat = trainer.train(train_dataloader, epoch=epoch, adversarial=False)
        train_loss, train_clean_acc = train_stat['loss'], train_stat['clean_acc']
        logger.log('Training Loss: {:.4f}.\tLR: {:.4f}'.format(train_loss, last_lr))
        
        test_clean_acc = trainer.eval(test_dataloader)
        val_clean_acc = trainer.eval(val_dataloader)
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tVal: {:.2f}%.\tTest: {:.2f}%.'.format(\
                        train_clean_acc*100, val_clean_acc*100, test_clean_acc*100))
        
        val_best, test_best = 0, 0
        epoch_metrics = {'train_loss':train_loss, 'train_clean_acc':train_clean_acc, 'train_adv_acc':0,\
                        'test_clean_acc':test_clean_acc, 'test_adv_acc':0,\
                        'val_clean_acc':val_clean_acc, 'val_adv_acc':0,\
                        'epoch': epoch, 'lr': last_lr} 
        
        
    # save latest checkpoint
    torch.save({'unaveraged_model': trainer.model.module.state_dict(),
               'wa_model': trainer.wa_model.module.state_dict(),
               'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.scheduler.state_dict(),
               'val_best':val_best, 'test_best':test_best, 'epoch':epoch}, \
               os.path.join(trainer.params.fname, 'latest_checkpoint.pt')) 
    
    if args.save_intermediate_models and epoch % args.save_intermediate_models == 0:
        if epoch > 0:
            torch.save({'unaveraged_model': trainer.model.module.state_dict(),
                        'wa_model': trainer.wa_model.module.state_dict(),
                   'optimizer': trainer.optimizer.state_dict(), 'scheduler': trainer.scheduler.state_dict(),
                   'val_best':val_best, 'test_best':test_best, 'epoch':epoch},\
                       os.path.join(trainer.params.fname, '{}.pt'.format(epoch))) 
    
    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    
    if epoch % 10 == 9:
        logger.log('\nCurrent Val_best: {:.2f}%\tTest_best: {:.2f}%\n'.format(val_best * 100,test_best * 100))
    
    metrics = pd.DataFrame(epoch_metrics, index=[0]) # each epoch, only hold metrics for this epoch and write to csv
    if epoch == 0:
        metrics.to_csv(os.path.join(args.fname, 'stats.csv'), mode='a', index=False, header=True)
    else:
        metrics.to_csv(os.path.join(args.fname, 'stats.csv'), mode='a', index=False, header=False)

logger.log('\nTraining completed. Val_best: {:.2f}%\tTest_best: {:.2f}%'.format(val_best * 100,test_best * 100))
