ARCH=$1
ID_dset=CIFAR-10

if [[ "$ID_dset" == "CIFAR-10" ]]; then
    if [[ "$ARCH" == "densenet" ]]; then
        LOAD_FILE=./checkpoints/cifar10/checkpoint_100.pth.tar
    elif [[ "$ARCH" == "WRN" ]]; then
        LOAD_FILE=./checkpoints/cifar10/cifar10_wrn_pretrained_epoch_99.pt
    fi
elif [[ "$ID_dset" == "CIFAR-100" ]]; then
    if [[ "$ARCH" == "densenet" ]]; then
        LOAD_FILE=./checkpoints/cifar100/checkpoint_100.pth.tar
    elif [[ "$ARCH" == "WRN" ]]; then
        LOAD_FILE=./checkpoints/cifar100/cifar100_wrn_pretrained_epoch_99.pt
    fi
else
    LOAD_FILE=None
fi
echo "==============extract features from $ID_dset ========="
CUDA_VISIBLE_DEVICES=1 python feat_cifar.py \
        --id_dset $ID_dset \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch $ARCH \
        --large_scale \
        --rerun \
        --test_bs 16 \
        --load_file ${LOAD_FILE}


ID_dset=CIFAR-100
if [[ "$ID_dset" == "CIFAR-10" ]]; then
    if [[ "$ARCH" == "densenet" ]]; then
        LOAD_FILE=./checkpoints/cifar10/checkpoint_100.pth.tar
    elif [[ "$ARCH" == "WRN" ]]; then
        LOAD_FILE=./checkpoints/cifar10/cifar10_wrn_pretrained_epoch_99.pt
    fi
elif [[ "$ID_dset" == "CIFAR-100" ]]; then
    if [[ "$ARCH" == "densenet" ]]; then
        LOAD_FILE=./checkpoints/cifar100/checkpoint_100.pth.tar
    elif [[ "$ARCH" == "WRN" ]]; then
        LOAD_FILE=./checkpoints/cifar100/cifar100_wrn_pretrained_epoch_99.pt
    fi
else
    LOAD_FILE=None
fi
#SVHN Texture LSUN_resize LSUN iSUN places365 \
echo "==============extract features from $ID_dset ========="
CUDA_VISIBLE_DEVICES=1 python feat_cifar.py \
        --id_dset $ID_dset \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch $ARCH \
        --large_scale \
        --rerun \
        --test_bs 16 \
        --load_file ${LOAD_FILE}