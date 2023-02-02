import os
import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np


DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 200)),
    'num_classes': 200,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}

# original code
# def load_tinyimagenet(data_dir, use_augmentation=False):
#     """
#     Returns Tiny Imagenet-200 train, test datasets and dataloaders.
#     Arguments:
#         data_dir (str): path to data directory.
#         use_augmentation (bool): whether to use augmentations for training set.
#     Returns:
#         train dataset, test dataset. 
#     """
#     test_transform = transforms.Compose([transforms.ToTensor()])
#     if use_augmentation:
#         train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), 
#                                               transforms.ToTensor()])
#     else: 
#         train_transform = test_transform
    
#     train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
#     test_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)

#     return train_dataset, test_dataset

# adapated code: add validation set
def load_tinyimagenet(data_dir, use_augmentation=False,validation=False):
    """
    Returns Tiny Imagenet-200 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, val dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    else:
        train_transform = test_transform
    
    np.random.seed(13) # seed = 13
    split_permutation = list(np.random.permutation(100000))
    
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
#     val_path = os.path.join(data_dir, 'test')
    
    if validation:
        train_dataset = torch.utils.data.Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
        val_dataset = torch.utils.data.Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
        test_dataset = ImageFolder(val_path, transform=test_transform)
        
        print('Using Tiny-ImageNet-200 with train size:{} val size:{} and test size:{}'.format(len(train_dataset),len(val_dataset),len(test_dataset)))

        return train_dataset, test_dataset, val_dataset
    
    else:
        train_dataset = ImageFolder(train_path, transform=train_transform)
        test_dataset = ImageFolder(val_path, transform=test_transform)
        return train_dataset, test_dataset