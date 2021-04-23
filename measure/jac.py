import numpy as np
import torch
from torch.nn import Conv1d, Conv2d, Linear
import datetime

def log(info, print_log=True):
    if print_log:
        print("[{}]  {}".format(datetime.datetime.now(), info), flush=True)

def get_slices(net):

    slices = []
    ind = 0
    for i, layer in enumerate(net.layers):
        slices.append([ind])
        for name, param in layer.named_parameters():
            ind += param.data.view(-1).shape[0]
        slices[-1].append(ind)
    return slices

def get_jacobian(train_loader, net, batchsize, num_classes, device=None):
    
    if device is None:
        device = train_loader.device

    batch_stack = 0
    slices = get_slices(net)
    total_dim = slices[-1][-1]
    break_ind = False
    grad_batch = torch.zeros([batchsize * num_classes, total_dim]).to(device)

    grad_count = 0

    for i, (input, target) in enumerate(iter(train_loader)):
        
        for cur_input, cur_target in zip(input, target):
            cur_input = cur_input.unsqueeze(0)
    
            for cur_lbl in range(num_classes):
                cur_gradient = []
                cur_one_hot = [0] * int(num_classes)
                cur_one_hot[cur_lbl] = 1
                cur_one_hot = torch.FloatTensor([cur_one_hot]).to(train_loader.device)
                
                net.zero_grad()
                cur_output = net(cur_input)
                cur_output.backward(cur_one_hot)
                
                for layer in net.layers:
                    for name, param in layer.named_parameters():
                        cur_gradient.append(param.grad.data.view(-1))
            
                grad_batch[grad_count] = torch.cat(cur_gradient).to(device)
                grad_count += 1
            if grad_count >= batchsize * num_classes:
                break_ind = True
                break
        if break_ind:
            break
    return grad_batch

def weighted_jacobian(jac, slices, weights):
    assert len(slices) == len(weights)
    assert jac.shape[-1] == slices[-1][-1]
    for i, (ind_s, ind_e) in enumerate(slices):
        jac[:, ind_s : ind_e] *= weights[i]
    

def get_jacobian_svd(train_loader, net, batchsize, num_classes, weights=None, device=None):
    log("Computing jacobian")
    slices = get_slices(net)
    jac = get_jacobian(train_loader, net, batchsize, num_classes, device)
    if weights is not None:
        weighted_jacobian(jac, slices, weights)
    log("Computing SVD of jacobian with shape {}".format(jac.shape))
    U, D, V = torch.svd_lowrank(jac, q=len(jac))
    
    splitted_norms = np.zeros([len(slices), len(D)])
    for i, v in enumerate(V.T):
        for j, [ind_s, ind_e] in enumerate(slices):
            d = ind_e - ind_s
            vec_partial = v[ind_s: ind_e]

            splitted_norms[j][i] = torch.norm(vec_partial)
    return D.cpu().numpy(), splitted_norms, slices, net.layer_names

class RunningStats:
    """Based on `John Cook's`__ method for computing the variance of the data using Welford's algorithm in single pass.

    __ http://www.johndcook.com/standard_deviation.html
    """
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())