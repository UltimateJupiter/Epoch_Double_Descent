import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from .ondevdl import OnDeviceDataLoader

def get_cifar_10(batch_size, ondev=True, ds_crop_train=1, ds_crop_test=1, device='cpu'):

    rgb_normalized_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    ds_train = CIFAR10(root='./data/cifar10', train=True, download=True, transform=rgb_normalized_transform)
    ds_test = CIFAR10(root='./data/cifar10', train=False, download=True, transform=rgb_normalized_transform)
    
    ds_train = crop_ds(ds_train, ds_crop=ds_crop_train)
    ds_test = crop_ds(ds_test, ds_crop=ds_crop_test)
    
    DL = OnDeviceDataLoader if ondev else DataLoader
    train_dl = DL(ds_train, batch_size, shuffle=True, device=device)
    test_dl = DL(ds_test, batch_size, shuffle=True, device=device)

    return train_dl, test_dl, [ds_train, ds_test]

def crop_ds(ds, ds_crop):

    assert ds_crop <= 1
    ind = int(len(ds) * ds_crop)
    remain = len(ds) - ind

    d1, d2 = random_split(ds, [ind, remain])
    return d1