
# settings
ARCH=$1
idset=$2
SCORE=${3}


# load file
if [[ $ARCH == "vit_b" ]]; then
    FEAT_NORM=1
else
    FEAT_NORM=0
fi



# get started
echo ""
echo ""
echo "=============================== CIFAR Benchmark: Method - ${SCORE}; ARCH - ${ARCH} =============================="

if [[ "$SCORE" == "KNN" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together \
        --feat_norm 1

elif [[ "$SCORE" == "ReAct" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "ModelClip" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "SSD" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "FDB" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "Line" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "VRA" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "DICEOOD" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "MahaVanilla" ]]; then
    python test_feat_ood_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --feat_norm ${FEAT_NORM}

elif [[ "$SCORE" == "Residual" ]]; then
    python test_logit_feat_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "VIM" ]]; then
    python test_logit_feat_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "Ige" ]]; then
    python test_logit_feat_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

elif [[ "$SCORE" == "Neco" ]]; then
    python test_feat_logit_ood.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets textures sun places inat imagenet_o \
        --arch ${ARCH} \
        --score ${SCORE} \
        --run_together

else
    python test_baseline_cifar.py \
        --id_dset $idset \
        --large_scale \
        --ood_dsets SVHN Texture LSUN_resize LSUN iSUN places365 \
        --arch ${ARCH} \
        --score ${SCORE} \
        --test_bs 8
fi