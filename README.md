## Are Diffusion Models Vulnerable to Membership Inference Attacks? [ICML 2023]
[arXiv](https://arxiv.org/abs/2302.01316)

This is the official implementation of the paper "Are Diffusion Models Vulnerable to Membership Inference Attacks?".
The proposed Step-wise Error Comparing Membership Inference (SecMI) is implemented in this codebase.

**[3/4/2024]** We have released [SecMI-LDM](https://github.com/jinhaoduan/SecMI-LDM): SecMI on Latent Diffusion Models. Since SecMI requires intermediate results, we modified the original diffuser and put all the data/checkpoints in another repo.

### Model Training
This codebase is built on top of [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm). 
Please follow its instructions for model training. You can also run the following commands (or refer to `train.sh`):
```shell
python main.py --train --logdir ./experiments/CIFAR10 \
--dataset CIFAR10 --img_size 32 --batch_size 128 --fid_cache ./stats/cifar10.train.npz --total_steps 800001
```
By default, it will load the splittings stored in `mia_evals/member_splits` and train DDPMs over half training split. 
You can specify `--dataset` and `--total_steps` as you want. 

### Pre-trained model

Some pre-trained models can be downloaded from [here](https://drexel0-my.sharepoint.com/:f:/g/personal/jd3734_drexel_edu/EnVid-empkpNvzC_mOfHwv0BpgkDsB_C4RmHO4rIH8BSzw?e=c17NjE).

### Run SecMI

To execute SecMI over pretrained DDPM, please execute the following command:

```python secmia.py --model_dir /path/to/model_dir --dataset_root /path/to/dataset --dataset cifar10 --t_sec 100 --k 10```

parameters:

- `--model_dir`: path to the directory of checkpoints 
- `--dataset_root`: path to the directory of datasets
- `--dataset`: dataset name
- `--t_sec`: timestep used for error comparing (`t_SEC` in the paper)
- `--k`: DDIM interval (`k` in the paper)

Please cite our paper if you feel this is helpful:
```
@InProceedings{duan2023are,
  title = {Are Diffusion Models Vulnerable to Membership Inference Attacks?},
  author = {Duan, Jinhao and Kong, Fei and Wang, Shiqi and Shi, Xiaoshuang and Xu, Kaidi},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {8717--8730},
  year = {2023}
}
```
