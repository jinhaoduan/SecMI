import os.path
import sys

# sys.path.append('.')

import torch
import torchvision.datasets
import numpy as np

# from measures import ssim
from score import fid


class ReconsDataset(torch.utils.data.Dataset):

    def __init__(self, data, transforms=None):
        super(ReconsDataset, self).__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image


class MIASTL10(torchvision.datasets.STL10):

    def __init__(self, idxs, **kwargs):
        super(MIASTL10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASTL10, self).__getitem__(item)


class MIACelebA(torchvision.datasets.CelebA):

    def __init__(self, idxs, **kwargs):
        super(MIACelebA, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACelebA, self).__getitem__(item)


class MIASVHN(torchvision.datasets.SVHN):

    def __init__(self, idxs, **kwargs):
        super(MIASVHN, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIASVHN, self).__getitem__(item)


class MIACIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR10, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR10, self).__getitem__(item)


class MIACIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, idxs, **kwargs):
        super(MIACIFAR100, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIACIFAR100, self).__getitem__(item)


class MIAImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, idxs, **kwargs):
        super(MIAImageFolder, self).__init__(**kwargs)
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = self.idxs[item]
        return super(MIAImageFolder, self).__getitem__(item)


def load_member_data(dataset_root, dataset_name, batch_size=128, member_split_root='./member_splits', shuffle=False, randaugment=False):
    if dataset_name.upper() == 'CIFAR10':
        splits = np.load(os.path.join(member_split_root, 'CIFAR10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        if randaugment:
            transforms = torchvision.transforms.Compose([torchvision.transforms.RandAugment(num_ops=5),
                                                         torchvision.transforms.ToTensor()])
        else:
            transforms = torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor()])
        member_set = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                transform=transforms)
        nonmember_set = MIACIFAR10(nonmember_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                   transform=transforms)
    elif dataset_name.upper() == 'CIFAR100':
        splits = np.load(os.path.join(member_split_root, 'CIFAR100_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        member_set = MIACIFAR100(member_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                 transform=transforms)
        nonmember_set = MIACIFAR100(nonmember_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                    transform=transforms)
    elif dataset_name.upper() == 'SVHN':
        splits = np.load(os.path.join(member_split_root, 'SVHN_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        member_set = MIASVHN(member_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                             transform=transforms)
        nonmember_set = MIASVHN(nonmember_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                                transform=transforms)
    elif dataset_name.upper() == 'CELEBA':
        splits = np.load(os.path.join(member_split_root, 'CELEBA_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        # load MIA Datasets
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIACelebA(member_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                               transform=transforms, download=True)
        nonmember_set = MIACelebA(nonmember_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                                  transform=transforms, download=True)
    elif dataset_name.upper() == 'STL10':
        splits = np.load(os.path.join(member_split_root, 'STL10_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'stl10'), split='train',
                              download=True, transform=transforms)
        nonmember_set = MIASTL10(nonmember_idxs, root=os.path.join(dataset_root, 'stl10'), split='train',
                                 download=True, transform=transforms)
    elif dataset_name.upper() == 'STL10-U':
        splits = np.load(os.path.join(member_split_root, 'STL10_Unlabeled_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                              download=True, transform=transforms)
        nonmember_set = MIASTL10(nonmember_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                                 download=True, transform=transforms)
    elif dataset_name.upper() == 'TINY-IN':
        splits = np.load(os.path.join(member_split_root, 'TINY-IN_train_ratio0.5.npz'))
        member_idxs = splits['mia_train_idxs']
        nonmember_idxs = splits['mia_eval_idxs']
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor()
        ])
        member_set = MIAImageFolder(member_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                    transform=transforms)
        nonmember_set = MIAImageFolder(nonmember_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                       transform=transforms)
    else:
        raise NotImplemented

    member_loader = torch.utils.data.DataLoader(member_set, batch_size=batch_size, shuffle=shuffle)
    nonmember_loader = torch.utils.data.DataLoader(nonmember_set, batch_size=batch_size, shuffle=shuffle)
    return member_set, nonmember_set, member_loader, nonmember_loader
