import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def get_cifar_10(batch_size):

    rgb_normalized_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    ds_train = CIFAR10(root='./data/cifar10', train=True, download=True, transform=rgb_normalized_transform)
    ds_test = CIFAR10(root='./data/cifar10', train=False, download=True, transform=rgb_normalized_transform)
    train_dl = DataLoader(ds_train, batch_size, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size, shuffle=True)

    return train_dl, test_dl, [ds_train, ds_test]