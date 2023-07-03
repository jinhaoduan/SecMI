CUDA_VISIBLE_DEVICES=0 python main.py --train --logdir ./experiments/CIFAR10 \
--dataset CIFAR10 --img_size 32 --fid_cache ./stats/CIFAR100.train.npz