import pdb

import torch
import numpy as np
import os
import random
from tqdm import tqdm

if __package__ is None:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import recall_level_default

def get_feat_dims(args, net, embed_mode=False,raw=False):
    if not args.large_scale:
        dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    else:
        res = 384 if ("vit" in args.arch and "clip" not in args.arch) else 224
        # res = 384 if ("vit" in args.arch or 'dino' in args.arch) else 224
        dummy_input = torch.zeros((1, 3, res, res)).cuda()

    # data type for vit_clip
    # if "clip" in args.arch and "vit" in args.arch:
        # dummy_input = dummy_input.half()
    
    with torch.no_grad():
        if raw:
            logits_or_featProj, feature_list = net.feature_list(dummy_input, raw=raw)
        else:
            logits_or_featProj, feature_list = net.feature_list(dummy_input)
    if raw:
        featdims = [[feat.shape[-2],feat.shape[-1]] for feat in feature_list]
    else:
        featdims = [feat.shape[1] for feat in feature_list]
    if args.arch == 'densenet':
        featdims = featdims[:-1]
    if embed_mode:
        feat_proj_dim = logits_or_featProj.shape[-1]
        return feat_proj_dim, np.array(featdims)
    else:
        return None, np.array(featdims)


def load_features_raw(args, id=False, ood_name=None, split="test", last_idx=None, \
                  featdim=None, projdim=None, train_random_num=None, feat_space=0):
    """THe featdims is for the np.memmap"""
    # feature save directory
    save_folder = os.path.join(args.save_folder, "raw_{}".format(args.id_dset))
    save_folder = os.path.join(save_folder, f"{args.arch}")

    # determine post name based on feature space
    if feat_space in [0]:
        post_name = None

    # supcon mode
    embed_mode = "supcon" in args.arch or "clip" in args.arch
    logit_or_projFeat_dim = projdim if embed_mode else args.num_classes

    if id:
        # parse file names
        file_name = f"{save_folder}/id_{split}_alllayers.npy"  # for small scale
        feat_file = f"{save_folder}/id_{split}_feat.mmap"  # for large scale
        label_file = f"{save_folder}/id_{split}_label.mmap"  # for large scale
        if embed_mode:
            logit_or_projFeat_file = f"{save_folder}/id_{split}_proj.mmap"
        else:
            logit_or_projFeat_file = f"{save_folder}/id_{split}_logit.mmap"  # for large scale

        # append train number for training feature
        if split == "train":
            feat_file = f"{feat_file.split('.')[0]}_{args.id_train_num}.mmap"
            label_file = f"{label_file.split('.')[0]}_{args.id_train_num}.mmap"
            logit_or_projFeat_file = f"{logit_or_projFeat_file.split('.')[0]}_{args.id_train_num}.mmap"
        if post_name is not None:
            feat_file = f"{feat_file.split('.')[0]}_{post_name}.mmap"

        # load
        if args.large_scale:
            assert os.path.exists(feat_file)
            print(f"Loading features from {feat_file}")
            feat = np.memmap(feat_file, dtype='float32', mode='r')
            logit_or_projFeat = np.memmap(logit_or_projFeat_file, dtype='float32', mode='r')
            label = np.memmap(label_file, dtype='float32', mode='r')
            feat = feat.reshape((-1, featdim[0, -2], featdim[0, -1]))
            logit_or_projFeat = logit_or_projFeat.reshape((-1, logit_or_projFeat_dim))
            label = label
        else:
            assert os.path.exists(file_name)
            print(f"Loading features from {file_name}")
            feat, logit, label = np.load(file_name, allow_pickle=True)
            feat = feat.T
            logit = logit.T
        # feat = feat.reshape(-1, 577, 768)
    else:
        # parse file names
        file_name = f"{save_folder}/ood_{ood_name}_alllayers.npy"  # for small scale
        feat_file = f"{save_folder}/ood_{ood_name}_feat.mmap"  # for large scale
        if embed_mode:
            logit_or_projFeat_file = f"{save_folder}/ood_{ood_name}_proj.mmap"
        else:
            logit_or_projFeat_file = f"{save_folder}/ood_{ood_name}_score.mmap"  # for large scale
        if post_name is not None:
            feat_file = f"{feat_file.split('.')[0]}_{post_name}.mmap"

        # load
        if args.large_scale:
            assert os.path.exists(feat_file)
            print(f"Loading features from {feat_file}")
            feat = np.memmap(feat_file, dtype='float32', mode='r')
            logit_or_projFeat = np.memmap(logit_or_projFeat_file, dtype='float32', mode='r')
            feat = feat.reshape((-1, featdim[0, -2], featdim[0, -1]))
            logit_or_projFeat = logit_or_projFeat.reshape((-1, logit_or_projFeat_dim))
        else:
            assert os.path.exists(file_name)
            print(f"Loading features from {file_name}")
            feat, logit = np.load(file_name, allow_pickle=True)
            feat = feat.T
            logit = logit.T

    # import pdb; pdb.set_trace()
    feat = np.ascontiguousarray(feat)
    logit_or_projFeat = np.ascontiguousarray(logit_or_projFeat)
    if id:
        label = np.ascontiguousarray(label)

    if not args.large_scale:
        feat = feat[:, :, last_idx:]

    if id:
        return feat, logit_or_projFeat, label
    else:
        return feat, logit_or_projFeat


def load_features_aug(args, id=False, ood_name=None, split="test", last_idx=None, \
                  featdim=None, projdim=None, train_random_num=None, feat_space=0, index=0):
    """THe featdims is for the np.memmap"""
    # feature save directory
    save_folder = os.path.join(args.save_folder, "{}".format(args.id_dset))
    save_folder = os.path.join(save_folder, f"{args.arch}")

    # determine post name based on feature space
    if feat_space in [0]:
        post_name = None

    # supcon mode
    embed_mode = "supcon" in args.arch or "clip" in args.arch or 'dino' in args.arch
    logit_or_projFeat_dim = projdim if embed_mode else args.num_classes

    if id:
        # parse file names
        file_name = f"{save_folder}/id_{split}_alllayers.npy"  # for small scale
        feat_file = f"{save_folder}/id_{split}_feat_aug.mmap"  # for large scale
        label_file = f"{save_folder}/id_{split}_label.mmap"  # for large scale
        if embed_mode:
            logit_or_projFeat_file = f"{save_folder}/id_{split}_proj.mmap"
        else:
            logit_or_projFeat_file = f"{save_folder}/id_{split}_logit.mmap"  # for large scale

        # append train number for training feature

        # if post_name is not None:
        #     feat_file = f"{feat_file.split('.')[0]}_{post_name}.mmap"

        # load
        if args.large_scale:
            print(f"Loading features from {feat_file}")
            assert os.path.exists(feat_file)

            feat = np.memmap(feat_file, dtype='float32', mode='r')
            logit_or_projFeat = np.memmap(logit_or_projFeat_file, dtype='float32', mode='r')
            label = np.memmap(label_file, dtype='float32', mode='r')
            feat = feat.reshape((-1,8, featdim))
            logit_or_projFeat = logit_or_projFeat.reshape((-1, logit_or_projFeat_dim))
            label = label
        else:
            assert os.path.exists(file_name)
            print(f"Loading features from {file_name}")
            feat, logit, label = np.load(file_name, allow_pickle=True)
            feat = feat.T
            logit = logit.T
        # feat = feat.reshape(-1, 577, 768)
    else:
        # parse file names
        file_name = f"{save_folder}/ood_{ood_name}_alllayers.npy"  # for small scale
        feat_file = f"{save_folder}/ood_{ood_name}_feat_aug.mmap"  # for large scale
        if embed_mode:
            logit_or_projFeat_file = f"{save_folder}/ood_{ood_name}_proj.mmap"
        else:
            logit_or_projFeat_file = f"{save_folder}/ood_{ood_name}_score.mmap"  # for large scale
        if post_name is not None:
            feat_file = f"{feat_file.split('.')[0]}_{post_name}.mmap"

        # load
        if args.large_scale:
            print(f"Loading features from {feat_file}")
            assert os.path.exists(feat_file)

            feat = np.memmap(feat_file, dtype='float32', mode='r')
            # print(feat.shape)
            logit_or_projFeat = np.memmap(logit_or_projFeat_file, dtype='float32', mode='r')
            feat = feat.reshape((-1,8, featdim))
            logit_or_projFeat = logit_or_projFeat.reshape((-1, logit_or_projFeat_dim))
        else:
            assert os.path.exists(file_name)
            print(f"Loading features from {file_name}")
            feat, logit = np.load(file_name, allow_pickle=True)
            feat = feat.T
            logit = logit.T

    # import pdb; pdb.set_trace()
    feat = np.ascontiguousarray(feat)
    logit_or_projFeat = np.ascontiguousarray(logit_or_projFeat)
    if id:
        label = np.ascontiguousarray(label)



    if id:
        return feat, logit_or_projFeat, label
    else:
        return feat, logit_or_projFeat
def load_features(args, id=False, ood_name=None, split="test", last_idx=None, \
        featdim=None, projdim=None, train_random_num=None, feat_space=0):
    """THe featdims is for the np.memmap"""
    # feature save directory
    save_folder = os.path.join(args.save_folder, "{}".format(args.id_dset))
    save_folder = os.path.join(save_folder, f"{args.arch}")

    # determine post name based on feature space
    if feat_space in [0]:
        post_name = None
    
    # supcon mode
    embed_mode = "supcon" in args.arch or "clip" in args.arch or 'dino' in args.arch
    logit_or_projFeat_dim = projdim if embed_mode else args.num_classes
    
    if id:
        # parse file names
        file_name = f"{save_folder}/id_{split}_alllayers.npy"   # for small scale
        feat_file = f"{save_folder}/id_{split}_feat.mmap"       # for large scale
        label_file = f"{save_folder}/id_{split}_label.mmap"     # for large scale
        if embed_mode:
            logit_or_projFeat_file = f"{save_folder}/id_{split}_proj.mmap"
        else:
            logit_or_projFeat_file = f"{save_folder}/id_{split}_logit.mmap"     # for large scale
        
        # append train number for training feature
        if split == "train":
            feat_file = f"{feat_file.split('.')[0]}_{args.id_train_num}.mmap"
            label_file = f"{label_file.split('.')[0]}_{args.id_train_num}.mmap"
            logit_or_projFeat_file = f"{logit_or_projFeat_file.split('.')[0]}_{args.id_train_num}.mmap"
        if post_name is not None:
            feat_file = f"{feat_file.split('.')[0]}_{post_name}.mmap"

        # load
        if args.large_scale:
            print(f"Loading features from {feat_file}")
            assert os.path.exists(feat_file)

            feat = np.memmap(feat_file, dtype='float32', mode='r')
            logit_or_projFeat = np.memmap(logit_or_projFeat_file, dtype='float32', mode='r')
            label = np.memmap(label_file, dtype='float32', mode='r')
            feat = feat.reshape((-1, featdim))
            logit_or_projFeat = logit_or_projFeat.reshape((-1, logit_or_projFeat_dim))
            label = label
        else:
            assert os.path.exists(file_name)
            print(f"Loading features from {file_name}")
            feat, logit, label = np.load(file_name, allow_pickle=True)
            feat = feat.T
            logit = logit.T
        # feat = feat.reshape(-1, 577, 768)
    else:
        # parse file names
        file_name = f"{save_folder}/ood_{ood_name}_alllayers.npy"   # for small scale
        feat_file = f"{save_folder}/ood_{ood_name}_feat.mmap"       # for large scale
        if embed_mode:
            logit_or_projFeat_file = f"{save_folder}/ood_{ood_name}_proj.mmap"
        else:
            logit_or_projFeat_file = f"{save_folder}/ood_{ood_name}_score.mmap"     # for large scale
        if post_name is not None:
            feat_file = f"{feat_file.split('.')[0]}_{post_name}.mmap"

        # load
        if args.large_scale:
            print(f"Loading features from {feat_file}")
            assert os.path.exists(feat_file)

            feat = np.memmap(feat_file, dtype='float32', mode='r')
            # print(feat.shape)
            logit_or_projFeat = np.memmap(logit_or_projFeat_file, dtype='float32', mode='r')
            feat = feat.reshape((-1, featdim))
            logit_or_projFeat = logit_or_projFeat.reshape((-1, logit_or_projFeat_dim))
        else:
            assert os.path.exists(file_name)
            print(f"Loading features from {file_name}")
            feat, logit = np.load(file_name, allow_pickle=True)
            feat = feat.T
            logit = logit.T
    
    # import pdb; pdb.set_trace()
    feat = np.ascontiguousarray(feat)
    logit_or_projFeat = np.ascontiguousarray(logit_or_projFeat)
    if id:
        label = np.ascontiguousarray(label)

    if not args.large_scale:
        feat = feat[:, :,last_idx:]
    
    if id:
        return feat, logit_or_projFeat, label
    else:
        return feat, logit_or_projFeat


def _to_np(input):
    """
    Will convert any data to numpy array
    """
    if isinstance(input, torch.Tensor):
        return input.data.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    elif isinstance(input, list):
        return np.array(input)
    else:
        raise NotImplementedError
    
def _to_tensor(data, device=None, half=False):
    if isinstance(data, torch.Tensor):
        if half: return data.to(device=device).half()
        else: return data.to(device=device).float()
    elif isinstance(data, np.ndarray):
        if half: torch.from_numpy(data).to(device=device).half()
        else: return torch.from_numpy(data).to(device=device).float()
    elif isinstance(data, list):
        if half: torch.tensor(data).to(device=device).half()
        else: return torch.tensor(data).to(device=device).float()
    else:
        raise NotImplementedError


class AverageMeter(object):
    """Computes and stores the average and current value.
    Update to prevent overflow in the average calculation
    Examples::
        >>> # Initialize a meter to record loss
        >>> avger = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> avger.update(value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.avg = self.avg * float(self.count) / float(self.count + n) + val * float(n) / float(self.count + n) 
        self.count += n


def save_model(model, optimizer, scheduler, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def print_measures(auroc, fpr, aupr_in=None, aupr_out=None, method_name='Ours', recall_level=recall_level_default):
    print('\t\t\t\t' + method_name)
    print('FPR{:d}:  \t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:  \t\t\t{:.2f}'.format(100 * auroc))

    if aupr_in is not None:
        print('AUPRIN: \t\t\t{:.2f}'.format(100 * aupr_in))
    if aupr_out is not None:
        print('AUPROUT:\t\t\t{:.2f}'.format(100 * aupr_out))
def select_features(train_features, train_logits, id_labels, ratio):
    num_class = int(id_labels.max() + 1)
    # N, D = train_features.shape
    shapelen = len(train_features.shape)
    print("read label shape is {}".format(id_labels.shape))
    features = []
    labels = []
    logits = []
    for i in tqdm(range(num_class)):
        # fetch out the data belongs to this class
        class_idx = np.where(id_labels == i)[0]

        selectnum = int(len(class_idx)*ratio)
        if selectnum<1: selectnum=1
        selectid = random.sample(range(0,len(class_idx)),selectnum)
        class_idx = class_idx[selectid]
        # print(class_idx,)
        if shapelen>2:
            cls_feature = train_features[class_idx, :,:]
        else:
            cls_feature = train_features[class_idx,:]
        cls_logit = train_logits[class_idx, :]
        if selectnum==1:
            if shapelen>2:
                cls_feature = cls_feature[None,:,:]
            else:
                cls_feature=cls_feature[None,:]
            cls_logit = cls_logit[None,:]
        # print(cls_feature.shape,selectnum)
        features.append(cls_feature)
        logits.append(cls_logit)
        labels.append(id_labels[class_idx])
    # print(train_features[0].shape)
    # print(labels)
    features, labels, logits = np.concatenate(features), np.concatenate(labels), np.concatenate(logits)
    print("selected features size is {}".format(features.shape))
    return features,labels,logits




if __name__ == '__main__':
    avg = AverageMeter()

    for i in range(3):
        avg.update(val=i+1, n=i+1)

        print('iter {}'.format(i))
        print(avg.avg)
