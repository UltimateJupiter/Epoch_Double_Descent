import torch
import numpy as np

def err_onehot_int(onehot, label):
    assert len(onehot) == len(label)
    return 1 - torch.mean((torch.argmax(onehot, -1) == label).float())