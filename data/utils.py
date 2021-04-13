import torch
import numpy as np

class FakeDL():

    def __init__(self, X, Y, device):
        self.device = device
        self.batchsize = len(X)
        self.X = X.to(device)
        self.Y = Y.to(device)
        self.active = True
        return
    
    def __len__(self):
        return 1
    
    def __next__(self):
        if not self.active:
            raise StopIteration
        self.active = False
        return [self.X, self.Y]
    
    def __iter__(self):
        self.active = True
        return self

    next = __next__