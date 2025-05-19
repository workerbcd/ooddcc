import torch
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import utils.SVHN_loader as svhn
import numpy as np
from tqdm import tqdm

from dataset.dataloaders import get_loaders_for_ood
from dataset.cifar import Cifar10, Cifar100
from models.get_models import get_model
from utils.argparser import FeatExtractArgs
from utils.utils import get_feat_dims

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# transform_test_largescale = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])

def get_cifar_ood_loaders(args):
    root = 'OOD-detection/'
    ood_test_loaders = []
    ood_test_dsets = []
    for ood_name in args.ood_dsets:
        print(ood_name)
        name = ood_name.lower()
        tr = transform
        # if name == 'places365':
        #     tr = transform_test_largescale
        # else:
        #     tr = transform
        if name == 'svhn':
            root_ = os.path.join(root, name)
            print(root_)
            ood_set = svhn.SVHN(root_, split='test', transform=tr, download=False)
        elif name== "cifar-10":
            ood_set = Cifar10(root='OOD-detection/cifar10', train=False, download=not dld,
                          transform=tr)
        elif name== "cifar-100":
            ood_set = Cifar100(root='OOD-detection/cifar100', train=False, download=not dld,
                              transform=tr)
        else:
            root_ = os.path.join(root, ood_name)
            if name == 'texture':
                root_ = os.path.join(root, 'dtd/images')
            ood_set = torchvision.datasets.ImageFolder(root=root_, transform=tr)

        ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=args.prefetch, pin_memory=True)
        ood_test_loaders.append(ood_loader)
        ood_test_dsets.append(ood_set)
    return ood_test_loaders, ood_test_dsets


argparser = FeatExtractArgs()
args = argparser.get_args()

if args.id_dset == "CIFAR-10":
    dld = os.path.exists('OOD-detection/cifar10/cifar-10-python')
    id_train_set = Cifar10(root='OOD-detection/cifar10', train=True, download=not dld,
                           transform=transform, sample_num=200000)
    id_test_set = Cifar10(root='OOD-detection/cifar10', train=False, download=not dld,
                          transform=transform)
    id_train_loader = torch.utils.data.DataLoader(id_train_set, batch_size=args.test_bs, shuffle=True,
                                                  num_workers=args.prefetch, pin_memory=True)
    id_test_loader = torch.utils.data.DataLoader(id_test_set, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=args.prefetch, pin_memory=True)
    num_classes = 10

elif args.id_dset == "CIFAR-100":
    dld = os.path.exists('OOD-detection/cifar100/cifar-100-python')
    id_train_set = Cifar100(root='OOD-detection/cifar100', train=True, download=not dld,
                            transform=transform, sample_num=2000000)
    id_test_set = Cifar100(root='OOD-detection/cifar100', train=False, download=not dld,
                           transform=transform)
    id_train_loader = torch.utils.data.DataLoader(id_train_set, batch_size=args.test_bs, shuffle=True,
                                                  num_workers=args.prefetch, pin_memory=True)
    id_test_loader = torch.utils.data.DataLoader(id_test_set, batch_size=args.test_bs, shuffle=True,
                                                 num_workers=args.prefetch, pin_memory=True)
    num_classes = 100

ood_loaders, _ = get_cifar_ood_loaders(args)

# embed_mode = "supcon" in args.arch or "clip" in args.arch
net = get_model(arch=args.arch, args=args, load_file=args.load_file, strict=True)
net.eval()
if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
elif args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_mode=False
cudnn.benchmark = True  # fire on all cylinders
proj_dim, featdims = get_feat_dims(args, net, embed_mode=False)

save_folder = os.path.join(args.save_folder, args.id_dset)
save_folder = os.path.join(save_folder, f"{args.arch}")
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# [('train', id_train_loader)]:#,/
for split, in_loader in [ ('train', id_train_loader),('test', id_test_loader)]:
    # for split, in_loader in [('test', id_test_loader)]:
    print(f"Extracting the features of {args.id_dset}-{split}...")
    # break
    data_num = len(in_loader.dataset)

    if args.large_scale and args.rerun:
        feat_log_name = f"{save_folder}/id_{split}_feat.mmap"
        label_log_name = f"{save_folder}/id_{split}_label.mmap"
        logit_or_projFeat_log_name = f"{save_folder}/id_{split}_proj.mmap" if embed_mode else f"{save_folder}/id_{split}_logit.mmap"
        if split == "train":
            feat_log_name = f"{feat_log_name.split('.')[0]}_{args.id_train_num}.mmap"
            label_log_name = f"{label_log_name.split('.')[0]}_{args.id_train_num}.mmap"
            logit_or_projFeat_log_name = f"{logit_or_projFeat_log_name.split('.')[0]}_{args.id_train_num}.mmap"
        feat_log = np.memmap(feat_log_name, dtype='float32', mode='w+', shape=(data_num, featdims[-1]))
        label_log = np.memmap(label_log_name, dtype='float32', mode='w+', shape=(data_num,))
        logit_or_projFeat_log = np.memmap(logit_or_projFeat_log_name, dtype='float32', mode='w+',
                                          shape=(data_num, proj_dim)) if embed_mode \
            else np.memmap(logit_or_projFeat_log_name, dtype='float32', mode='w+', shape=(data_num, args.num_classes))
    else:
        feat_log = np.zeros((data_num, sum(featdims)))
        logit_log = np.zeros((data_num, args.num_classes))
        label_log = np.zeros(data_num)
        save_name = f"{save_folder}/id_{split}_alllayers.npy"

    if args.rerun:
        net.eval()
        with torch.no_grad():
            with tqdm(total=data_num) as pbar:
                for batch_idx, (inputs, targets) in enumerate(in_loader):
                    # import pdb; pdb.set_trace()
                    # if batch_idx >= 1:
                    # break
                    inputs, targets = inputs.to(device), targets.to(device)
                    start_ind = batch_idx * args.test_bs
                    end_ind = min((batch_idx + 1) * args.test_bs, len(in_loader.dataset))

                    # NOTE: by default the supcon model WON'T normalize feature
                    logit_or_projFeat, feature_list = net.feature_list(inputs)

                    # save all features for small scale, but last feature for large scale
                    feat = feature_list[-1]
                    if "resnet50" in args.arch:
                        feat = F.adaptive_avg_pool2d(feature_list[-1], 1).squeeze()

                    feat_log[start_ind:end_ind, :] = feat.data.cpu().numpy()
                    label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                    logit_or_projFeat_log[start_ind:end_ind] = logit_or_projFeat.data.cpu().numpy()

                    pbar.update(args.test_bs)

                    # if end_ind >= data_num:
                    #     break

            if not args.large_scale:
                np.save(save_name, (feat_log.T, logit_log.T, label_log))
    else:
        if args.large_scale:
            feat_log = np.memmap(f"{save_folder}/id_{split}_feat_{args.id_train_num}.mmap", dtype='float32', mode='r',
                                 shape=(len(in_loader.dataset), featdims[-1]))
            logit_log = np.memmap(f"{save_folder}/id_{split}_logit_{args.id_train_num}.mmap", dtype='float32', mode='r',
                                  shape=(len(in_loader.dataset), args.num_classes))
            label_log = np.memmap(f"{save_folder}/id_{split}_label_{args.id_train_num}.mmap", dtype='float32', mode='r',
                                  shape=(len(in_loader.dataset),))
        else:
            feat_log, logit_log, label_log = np.load(save_name, allow_pickle=True)
            feat_log, logit_log = feat_log.T, logit_log.T
# assert 1==2
# ====================== Save features for the OOD data
for dset_name, out_loader in zip(args.ood_dsets,ood_loaders):

    # dset_name = out_loader.dataset.name

    # only sample at most the same amount of data in the ID test set
    # if len(out_loader.dataset) > len(id_test_loader.dataset):
    #     # print(f"Too many samples. Will sample {len(id_test_loader.dataset)} out of {len(out_loader.dataset)}")
    #     # num_samples = len(id_test_loader.dataset)
    #     num_samples = len(out_loader.dataset)
    #     print(dset_name)
    # else:
    #     # num_samples = len(out_loader.dataset)
    #     continue
    # if dset_name != 'places365': continue
    num_samples = len(out_loader.dataset)
    # allocate memory
    if args.large_scale:
        ood_feat_log_file = f"{save_folder}/ood_{dset_name}_feat.mmap"
        ood_logit_or_projFeat_log_file = f"{save_folder}/ood_{dset_name}_proj.mmap" if embed_mode \
            else f"{save_folder}/ood_{dset_name}_score.mmap"
        if os.path.exists(ood_feat_log_file) and not args.rerun:
            print(f"Features of {dset_name} already exist. Going to the next...")
            continue
        else:
            ood_feat_log = np.memmap(ood_feat_log_file, dtype='float32', mode='w+', shape=(num_samples, featdims[-1]))
            ood_logit_or_projFeat_log = np.memmap(ood_logit_or_projFeat_log_file, dtype='float32', mode='w+',
                                                  shape=(num_samples, proj_dim)) if embed_mode \
                else np.memmap(ood_logit_or_projFeat_log_file, dtype='float32', mode='w+',
                               shape=(num_samples, args.num_classes))
    else:
        save_name = f"{save_folder}/ood_{dset_name}_alllayers.npy"
        if os.path.exists(save_name) and not args.rerun:
            print(f"Features of {dset_name} already exist. Going to the next...")
            continue
        else:
            ood_feat_log = np.zeros((num_samples, sum(featdims)))
            ood_logit_log = np.zeros((num_samples, args.num_classes))

    # run
    print(f"\n Extracting the features of {dset_name}...")

    # get start
    net.eval()
    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            for batch_idx, (inputs, _) in enumerate(out_loader):
                # if batch_idx >= 1:
                # break
                inputs = inputs.to(device)
                start_ind = batch_idx * args.test_bs
                end_ind = min((batch_idx + 1) * args.test_bs, num_samples)

                # NOTE: by default the supcon model WON'T normalize feature
                logit_or_projFeat, feature_list = net.feature_list(inputs)

                # save all features for small scale, but last feature for large scale
                feat = feature_list[-1]
                if "resnet50" in args.arch:
                    feat = F.adaptive_avg_pool2d(feature_list[-1], 1).squeeze()

                ood_feat_log[start_ind:end_ind, :] = feat.data.cpu().numpy()[:end_ind - start_ind, :]
                ood_logit_or_projFeat_log[start_ind:end_ind] = logit_or_projFeat.data.cpu().numpy()[
                                                               :end_ind - start_ind]

                pbar.update(args.test_bs)

                # quit if sampled enough data
                if end_ind >= num_samples:
                    break

        if not args.large_scale:
            np.save(save_name, (ood_feat_log.T, ood_logit_log.T))