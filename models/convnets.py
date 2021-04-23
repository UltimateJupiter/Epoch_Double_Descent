import torch.nn as nn
from .utils import get_children, get_layer_names

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Conv4FC1(nn.Module): 

    def __init__(self, inp_ch=3, hid_ch=64, out_ch=10, n_layers=5, bn=True): 

        super().__init__()
        self.inp_ch = inp_ch
        self.hid_ch = hid_ch
        self.out_ch = out_ch

        self.classifier = self.make_layer(bn)
        
        self.layers = get_children(self)
        self.name = 'Conv4FC1' + '_bn{}'.format(int(bn))

        self.layer_names = get_layer_names(self)

    def make_layer(self, bn):
        num_planes = self.hid_ch

        if bn:
            return nn.Sequential(
                # Layer 0
                nn.Conv2d(self.inp_ch, num_planes, kernel_size=3, stride=1,
                        padding=1, bias=True),
                nn.BatchNorm2d(num_planes),
                nn.ReLU(),

                # Layer 1
                nn.Conv2d(num_planes, num_planes*2, kernel_size=3,
                        stride=1, padding=1, bias=True),
                nn.BatchNorm2d(num_planes*2),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Layer 2
                nn.Conv2d(num_planes*2, num_planes*4, kernel_size=3,
                        stride=1, padding=1, bias=True),
                nn.BatchNorm2d(num_planes*4),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Layer 3
                nn.Conv2d(num_planes*4, num_planes*8, kernel_size=3,
                        stride=1, padding=1, bias=True),
                nn.BatchNorm2d(num_planes*8),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Layer 4
                nn.MaxPool2d(4),
                Flatten(),
                nn.Linear(num_planes*8, self.out_ch, bias=True)
            )
        else:
            return nn.Sequential(
                # Layer 0
                nn.Conv2d(self.inp_ch, num_planes, kernel_size=3, stride=1,
                        padding=1, bias=True),
                nn.ReLU(),

                # Layer 1
                nn.Conv2d(num_planes, num_planes*2, kernel_size=3,
                        stride=1, padding=1, bias=True),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Layer 2
                nn.Conv2d(num_planes*2, num_planes*4, kernel_size=3,
                        stride=1, padding=1, bias=True),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Layer 3
                nn.Conv2d(num_planes*4, num_planes*8, kernel_size=3,
                        stride=1, padding=1, bias=True),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Layer 4
                nn.MaxPool2d(4),
                Flatten(),
                nn.Linear(num_planes*8, self.out_ch, bias=True)
            )

    def forward(self, x): 
        out = self.classifier(x)
        return out

def make_cnn(num_planes=64, num_classes=10):
    ''' Returns a 5-layer CNN with width parameter c. '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, num_planes, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(num_planes),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(num_planes, num_planes*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(num_planes*2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(num_planes*2, num_planes*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(num_planes*4),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(num_planes*4, num_planes*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(num_planes*8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(num_planes*8, num_classes, bias=True)
    )