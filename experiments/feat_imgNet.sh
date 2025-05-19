ARCH=$1
gpu=$2
LOAD_FILE=None




CUDA_VISIBLE_DEVICES=$gpu python feat_extraction.py \
        --id_dset imagenet \
        --ood_dsets textures sun places inat imagenet_o openimage_o \
        --arch $ARCH \
        --large_scale \
        --id_train_num 2000000 \
        --id_train_num 200000 \
        --rerun \
        --test_bs 64 \
        --load_file ${LOAD_FILE}

