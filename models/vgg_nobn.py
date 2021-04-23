'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from .utils import get_children, get_layer_names

cfg = {
    'VGG11_nobn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11N_nobn': [32, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11NN_nobn': [16, 'M', 32, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGG13_nobn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16_nobn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19_nobn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_nobn(nn.Module):
    def __init__(self, vgg_name, n_classes=10):
        super(VGG_nobn, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(cfg[vgg_name][-2], n_classes)
        self.layers = get_children(self)
        self.name = vgg_name
        self.layer_names = get_layer_names(self)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11_nobn():
    return VGG_nobn('VGG11_nobn')

def VGG11N_nobn():
    return VGG_nobn('VGG11N_nobn')

def VGG11NN_nobn():
    return VGG_nobn('VGG11NN_nobn')

def VGG13_nobn():
    return VGG_nobn('VGG13_nobn')

def VGG16_nobn():
    return VGG_nobn('VGG16_nobn')

def VGG19_nobn():
    return VGG_nobn('VGG19_nobn')
