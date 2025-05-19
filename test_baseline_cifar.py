import pdb
import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import utils.SVHN_loader as svhn
from dataset.cifar import Cifar10, Cifar100


from utils.argparser import OODArgs
from dataset.get_datasets import NAMES
from dataset.dataloaders import get_loaders_for_ood
from models.get_models import get_model
from ood_scores.get_scorers import get_scorer
from utils.evaluator import Evaluator


# arguments
argparser = OODArgs()
args = argparser.get_args()

# ========================== Prepare
# ------ data loaders
transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
def get_cifar_ood_loaders(args):
    root = 'OOD-detection/'
    ood_test_loaders = []
    ood_test_dsets = []
    for ood_name in args.ood_dsets:
        print(ood_name)
        name = ood_name.lower()
        if name == 'svhn':
            root_ = os.path.join(root,name)
            ood_set = svhn.SVHN(root_, split='test', transform=transform, download=False)
        elif name == "cifar-10":
            ood_set = Cifar10(root='OOD-detection/cifar10', train=False, download=not dld,
                              transform=transform)
        elif name == "cifar-100":
            ood_set = Cifar100(root='OOD-detection/cifar100', train=False, download=not dld,
                               transform=transform)
        else:
            root_ = os.path.join(root,ood_name)
            if name == 'texture':
                root_ = os.path.join(root, 'dtd/images')
            ood_set = torchvision.datasets.ImageFolder(root=root_,transform=transform)

        ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
        ood_test_loaders.append(ood_loader)
        ood_test_dsets.append(ood_set)
    return ood_test_loaders, ood_test_dsets


if args.id_dset == "CIFAR-10":
    dld = os.path.exists('OOD-detection/cifar10/cifar-10-python')
    id_train_set = Cifar10(root='OOD-detection/cifar10', train=True, download=not dld, transform=transform, sample_num=20000)
    id_test_set = Cifar10(root='OOD-detection/cifar10', train=False, download=not dld, transform=transform)
    id_train_loader = torch.utils.data.DataLoader(id_train_set, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
    id_test_loader = torch.utils.data.DataLoader(id_test_set, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
    num_classes = 10

elif args.id_dset == "CIFAR-100":
    dld = os.path.exists('OOD-detection/cifar100/cifar-100-python')
    id_train_set = Cifar100(root='OOD-detection/cifar100', train=True, download=not dld, transform=transform, sample_num=20000)
    id_test_set = Cifar100(root='OOD-detection/cifar100', train=False, download=not dld, transform=transform)
    id_train_loader = torch.utils.data.DataLoader(id_train_set, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
    id_test_loader = torch.utils.data.DataLoader(id_test_set, batch_size=args.test_bs, shuffle=True,
                                num_workers=args.prefetch, pin_memory=True)
    num_classes = 100

ood_loaders, _ = get_cifar_ood_loaders(args)
ood_num = len(id_test_loader.dataset)
print("Dataloaders ready. \n")

# ------ model. Enforce strict here.
print(f"Number of ID classes: {args.num_classes}")
if args.id_dset =='CIFAR-100':
    if args.arch =='densenet':
        ldfile ='checkpoints/cifar100/checkpoint_100.pth.tar'
    elif args.arch =='WRN':
        ldfile = './checkpoints/cifar100/cifar100_wrn_pretrained_epoch_99.pt'
elif args.id_dset == 'CIFAR-10':
    if args.arch == 'densenet':
        ldfile= 'checkpoints/cifar10/checkpoint_100.pth.tar'
    elif args.arch == 'WRN':
        ldfile = './checkpoints/cifar10/cifar10_wrn_pretrained_epoch_99.pt'
else:
    raise NameError
net = get_model(arch=args.arch, args=args, load_file=ldfile, strict=True)
net.eval()
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
elif args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

cudnn.benchmark = True  # fire on all cylinders

# ------ scorer
scorer = get_scorer(args.score, args)
if args.score in ["Maha", "ODIN"]:
    scorer.fit(net)
else:
    scorer.fit()

# evaluator
evaluator = Evaluator(args, ood_num=ood_num)

# =========================== Get started
# ID test set
print(f"\n\n Calculating the scores for the ID test set...")
evaluator.eval(net, id_test_loader, scorer, in_dist=True)

# OODs
ood_names = [name.lower() for name in args.ood_dsets]
for ood_name, out_loader in zip(ood_names,ood_loaders):
    print(f"\n\nEvaluating on {args.id_dset} v.s. {ood_name}...")
    evaluator.eval(net, out_loader, scorer, in_dist=False)

# Mean results
evaluator.print_mean_results()
# ouput_excel(args,'cifar.xlsx', ood_names, aurocs,fprs)