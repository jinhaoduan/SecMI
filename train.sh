
CUDA_VISIBLE_DEVICES=0 python main.py --train --logdir ./experiments/CIFAR10 \
--dataset CIFAR10 --img_size 32 --batch_size 128 --fid_cache ./stats/cifar10.train.npz --total_steps 800001
