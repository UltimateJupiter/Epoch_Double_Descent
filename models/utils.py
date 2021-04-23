import torch
from torch.nn import Conv1d, Conv2d, Linear
import numpy as np

def init_network(net, scales=None, init=None):

    if scales is None:
        scales = [1] * len(net.layers)
    else:
        assert len(scales) == len(net.layers)
    
    if init is None:
        init = 'X' * len(net.layers)
    else:
        assert len(init) == len(net.layers)

    for i, layer in enumerate(net.layers):
        with torch.no_grad():
            if init[i] == 'N':
                torch.nn.init.kaiming_normal_(layer.weight, a=np.sqrt(5))
            elif init[i] == 'U':
                torch.nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
            elif init[i] == 'X':
                torch.nn.init.xavier_normal_(layer.weight)
            else:
                raise Exception
            layer.weight *= scales[i]
    print(scales)
    return net

def get_children(model: torch.nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    ret_children = []
    for child in flatt_children:
        if isinstance(child, (Conv1d, Conv2d, Linear)):
            ret_children.append(child)

    return ret_children

def get_layer_names(model):
    layer_names = []

    conv_ind = 1
    fc_ind = 1

    for layer in model.layers:
        if isinstance(layer, (Conv1d, Conv2d)):
            layer_names.append('Conv{}'.format(conv_ind))
            conv_ind += 1
        elif isinstance(layer, Linear):
            layer_names.append('FC{}'.format(fc_ind))
            fc_ind += 1
    
    return layer_names
            