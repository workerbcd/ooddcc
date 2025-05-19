arch=$1 #densenet, WRN
id_dset=CIFAR-100

bash experiments/run_cifar_clf_final.sh $arch $id_dset Dynamic



id_dset=CIFAR-10

bash experiments/run_cifar_clf_final.sh $arch $id_dset Dynamic

