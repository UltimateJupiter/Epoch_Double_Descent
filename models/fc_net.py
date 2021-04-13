import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

import numpy as np
from .utils import get_children


class FC2(nn.Module):

    def __init__(self, inp, hid, out): 

        super().__init__()

        self.fc1 = nn.Linear(inp, hid) 
        self.fc2 = nn.Linear(hid, out)

        self.layers = get_children(self)
        self.name = "FC2"

    def forward(self, x): 

        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out