import argparse
import numpy as np
import torch

from models.resnet import resnet18, resnet34, resnet50, resnet101
from models.vit import ViT

from models.densenet import DenseNet3
from models.WideResnet import WideResNet
from models.swin import swin_v2_b
from models.deit import deit_base_patch16_224
from models.dino import vit_base


torch.hub.set_dir("./.cache")
def get_model(arch, args:argparse.Namespace, load_file=None, strict=True):
    """
    Get model

    Arch:
        arch (str):             Architecture name
        args (namespace):       Might contain network settings (e.g. layer number)
        load_file(str):         The parameter file to load. If None, will randomize the parameter
        stric(str):             Strict network parameter loading. N/A if load_file is None
    Return:
        net:                    The neural network model
    """
    arch = arch.lower()

    # create model
    load_pretrain = load_file is None
    if arch == 'resnet50':
        net = resnet50(pretrained=load_pretrain, num_classes=args.num_classes)
    elif arch == "vit_b":
        net = ViT('B_16_imagenet1k', pretrained=True)
    elif arch == "densenet":
        net = DenseNet3(100, args.num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None)
    elif arch == 'wrn':
        net = WideResNet(40,args.num_classes,2,0.5)
    elif arch == 'swin_b':
        net = swin_v2_b(pretrained=True)
    elif arch == 'deit':
        net = deit_base_patch16_224(pretrained=True)
    elif arch == 'dino':
        net = vit_base(16)
        chepoint = torch.load('.cache/checkpoints/dino_vitbase16_pretrain.pth')
        net.load_state_dict(chepoint,strict=True)
    else:
        raise NotImplementedError(f"The architecture {arch} is not implemented.")

    # load parameter
    # import pdb; pdb.set_trace()
    if load_file is not None:
        # trim state_dict keys for models trained on multi-gpus
        if "supcon" in arch or 'simclr' in arch:
            state_dict = torch.load(load_file)["model"]
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 3:
                    net.encoder = torch.nn.DataParallel(net.encoder)
                else:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        k = k.replace("module.", "")
                        k = k.replace("encoder.", "")
                        new_state_dict[k] = v
                    state_dict = new_state_dict
            net.load_state_dict(state_dict, strict=strict)
        elif args.arch == 'densenet':
            ckpoint = torch.load(load_file)
            net.load_state_dict(ckpoint['state_dict'])
        else:
            print(load_file)
            state_dict = torch.load(load_file)
            net.load_state_dict(state_dict, strict=strict)
        
        # load
        print("Model loaded from: {}".format(load_file))
    
    return net
        
   