import os, sys
from functools import partial

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


if __package__ == None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.get_datasets import get_dataset
from dataset.aug_transforms import aug_type

def get_loaders_for_ood_aug(args, trf=None):
    """Get the ID train/test loader and all the OOD loaders
    This is for the OOD detection evaluation, in the sense that:
    1. The id_train and id_test set won't be shuffled, and the test transform will be applied
    2. The OOD dataloaders will use the id test transform, and will be shuffled for random selection.

    Args:
        trf (torchvision.Transforms):       The transform to apply on all datasets.
                                            If None, will use default test transforms for the ID dataset based on the architecture used.

    Returns:
        id_train_loader:            The indistribution train loader
        id_test_loader:             The indistribution test loader
        ood_test_loaders (list):    The list of OOD test loaders.
    """
    trans = []
    for i in range(8):
        if ("vit" in args.arch and "clip" not in args.arch):
            if i == 0:
                test_trans = None
            else:
                test_trans = transforms.Compose([transforms.Resize((384, 384)),
                                                 aug_type[i - 1],
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(0.5, 0.5),
                                                 ])
        else:
            if i == 0:
                test_trans = None
            else:
                test_trans = transforms.Compose([
                    transforms.Resize(256),
                    aug_type[i - 1],
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
        trans.append(test_trans)
    torch.manual_seed(0)
    id_loaders = []
    for tran in trans:
        id_test_dset = get_dataset(args.id_dset, split="test", trf=tran, trf_type="test", trf_idSet=args.id_dset,
                                   model=args.arch)

        id_test_loader = DataLoader(id_test_dset, batch_size=args.test_bs, shuffle=False,
                                    num_workers=args.prefetch, pin_memory=True)
        id_loaders.append(id_test_loader)

    ood_loaders = []
    for ood_name in args.ood_dsets:
        ood_test_loaders = []
        for tran in trans:
            ood_dset = get_dataset(name=ood_name, split="test", trf=tran, trf_type="test", trf_idSet=args.id_dset,
                                   model=args.arch)
            ood_loader = DataLoader(ood_dset, batch_size=args.test_bs, shuffle=False,
                                    num_workers=args.prefetch, pin_memory=True)
            ood_test_loaders.append(ood_loader)
        ood_loaders.append(ood_test_loaders)
    return None, id_loaders, ood_loaders


def get_loaders_for_ood(args, trf=None):
    """Get the ID train/test loader and all the OOD loaders 
    This is for the OOD detection evaluation, in the sense that:
    1. The id_train and id_test set won't be shuffled, and the test transform will be applied
    2. The OOD dataloaders will use the id test transform, and will be shuffled for random selection.

    Args:
        trf (torchvision.Transforms):       The transform to apply on all datasets. 
                                            If None, will use default test transforms for the ID dataset based on the architecture used.

    Returns:
        id_train_loader:            The indistribution train loader
        id_test_loader:             The indistribution test loader
        ood_test_loaders (list):    The list of OOD test loaders.
    """
    id_train_dset = get_dataset(args.id_dset, split="train", trf=trf, trf_type="test", trf_idSet=args.id_dset, data_sample_num=args.id_train_num, model=args.arch)
    id_test_dset = get_dataset(args.id_dset, split="test", trf=trf, trf_type="test", trf_idSet=args.id_dset, model=args.arch)
    id_train_loader = DataLoader(id_train_dset, batch_size=args.test_bs, shuffle=True, 
                                num_workers=args.prefetch, pin_memory=True)
    id_test_loader = DataLoader(id_test_dset, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)

    ood_test_loaders = []
    for ood_name in args.ood_dsets:
        ood_dset = get_dataset(name=ood_name, split="test", trf=trf, trf_type="test", trf_idSet=args.id_dset, model=args.arch)
        ood_loader = DataLoader(ood_dset, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
        ood_test_loaders.append(ood_loader)
    
    return id_train_loader, id_test_loader, ood_test_loaders
