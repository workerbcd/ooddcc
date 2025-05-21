# Improving Out-of-Distribution Detection via Dynamic Covariance Calibration
The paper has been accepted in ICML 2025

This code is implemented based on https://github.com/ivalab/WDiscOOD
## Preparetion

- dataset
  
  1. on CIFAR benchmarks
  ```  
  CIFAR10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -> OOD-detection/CIFAR-10
  CIFAR100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz  -> OOD-detection/CIFAR-100
  SVHN: http://ufldl.stanford.edu/housenumbers/test_32x32.mat -> OOD-detection/svhn, Then run `python select_svhn_data.py` to generate test subset.
  LSUN: https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz -> OOD-detection/LSUN
  LSUN-R: https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz -> OOD-detection/ood_datasets/LSUN_resize
  iSUN: https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz -> OOD-detection/iSUN
  Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -> OOD-detection/dtd
  Places365: http://data.csail.mit.edu/places/places365/test_256.tar -> OOD-detection/places365
  ```

  

  2. on ImageNet benchmark 
  ```
  ImageNet: http://www.image-net.org/challenges/LSVRC/2012/index (after login, using https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) -> ./dataset/ILSVRC-2012/val
  iNaturalist: http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz -> OOD-detection/iNaturalist
  SUN: http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz -> OOD-detection/SUN
  Places: http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz -> OOD-detection/Places
  Textures: https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -> OOD-detection/dtd
  ImageNet-O: Follow https://github.com/hendrycks/natural-adv-examples -> OOD-detection/imagenet-o
  OpenImage-O: Follow https://github.com/haoqiwang/vim -> OOD-detection/openimage_o
  ```

- model 
```
DenseNet: https://github.com/deeplearning-wisc/dice/tree/master/checkpoints
WideResNet: https://github.com/PeymanMorteza/GEM/tree/main/CIFAR/snapshots/pretrained
ResNet-50: https://download.pytorch.org/models/resnet50-19c8e357.pth
ViT: Follow https://github.com/lukemelas/PyTorch-Pretrained-ViT
Swin_b: https://download.pytorch.org/models/swin_v2_b-781e5279.pth
DeiT: https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
DINO: https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth -> ./.cache/dino_vitbase16_pretrain.pth
```
all the models are publicly released. 
For DenseNet and WideResNet, please manually download the models and place them according to the bash file ```experiments/feat_Cifar.sh```
For DINO, please manually download the model and place it to ```.cache/dino_vitbase16_pretrain.pth```
other models can be downloaded automatically in our code. 
## Run 
extract features
```bash
   bash experiments/feat_Cifar.sh
   bash experiments/feat_imgNet.sh
   ```

run methods
```bash
    bash experiments/run-all.sh
    bash experiments/run_all_cifar.sh
```

the code of our method is in the file  ```ood_scores/score_dynamic.py```
