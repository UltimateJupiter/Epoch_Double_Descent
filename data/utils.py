import torch
import numpy as np
import torch
import datetime
from queue import PriorityQueue
import numpy as np
import time
import itertools

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

def log(info, print_log=True):
    if print_log:
        print("[{}]  {}".format(datetime.datetime.now(), info))

def timer(st, info='', timer_on=True):
    if timer_on:
        print("[{}]  {} {}".format(datetime.datetime.now(), datetime.datetime.now() - st, info))

def est(st, progress, info=''):
    time_diff = datetime.datetime.now() - st
    est = (1 - progress) * time_diff / progress
    print("[{}]  {:.3g}%\t EST: {}".format(datetime.datetime.now(), progress * 100, str(est)[:-4]))

def get_tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()

def get_tensor_dict_size(tensor_dict, p=False):
    ret = 0
    for k in tensor_dict:
        ret += get_tensor_size(tensor_dict[k])
    if p:
        print("{:.4g} G".format(ret / (2 ** 30)))
    return ret

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device) # pylint: disable=no-member
    indices = torch.arange(result.numel(), device=device).reshape(shape) # pylint: disable=no-member
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def kmax_argsort(a, b, k, return_vals=False):

    q = PriorityQueue()
    la, lb = len(a), len(b)
    assert k <= la * lb
    
    q.put((- a[0] * b[0], (0, 0)))
    vals, args = [], []
    args_set = set((0, 0))

    for _ in range(k):
    
        val, (ia, ib) = q.get()
        vals.append(-val)
        args.append((ia, ib))
    
        if ia + 1 < la:
            if (ia + 1, ib) not in args_set:
                args_set.add((ia + 1, ib))
                q.put((- a[ia + 1] * b[ib], (ia + 1, ib)))
    
        if ib + 1 < lb:
            if (ia, ib + 1) not in args_set:
                args_set.add((ia, ib + 1))
                q.put((- a[ia] * b[ib + 1], (ia, ib + 1)))
    
    if return_vals:
        return args, vals
    else:
        return args

def kp_2d(t1, t2):
    t1_h, t1_w = t1.size()
    t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    tiled_t2 = t2.repeat(t1_h, t1_w)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_h, t2_w, 1)
          .view(out_h, out_w)
    )
    return expanded_t1 * tiled_t2

def bkp_2d_raw(t1, t2):
    btsize, t1_h, t1_w = t1.size()
    btsize, t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    tiled_t2 = t2.repeat(1, t1_h, t1_w)
    expanded_t1 = (
        t1.unsqueeze(3)
          .unsqueeze(4)
          .repeat(1, 1, t2_h, t2_w, 1)
          .view(btsize, out_h, out_w)
    )
    expanded_t1 *= tiled_t2
    return expanded_t1

def bkp_2d(t1, t2):
    btsize, t1_h, t1_w = t1.size()
    btsize, t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    expanded_t1 = (
        t1.unsqueeze(3)
          .unsqueeze(4)
          .repeat(1, 1, t2_h, t2_w, 1)
          .view(btsize, out_h, out_w)
    )
    for i in range(t1_h):
        for j in range(t1_w):
            expanded_t1[:, i * t2_h: (i + 1) * t2_h, j * t2_w: (j + 1) * t2_w] *= t2
    return expanded_t1


def eigenthings_tensor_utils(t, device=None, out_device='cpu', symmetric=False, topn=-1):
    t = t.to(device)
    if topn >= 0:
        _, eigenvals, eigenvecs = torch.svd_lowrank(t, q=min(topn, t.size()[0], t.size()[1]))
        eigenvecs.transpose_(0, 1)
    else:
        if symmetric:
            eigenvals, eigenvecs = torch.symeig(t, eigenvectors=True) # pylint: disable=no-member
            eigenvals = eigenvals.flip(0)
            eigenvecs = eigenvecs.transpose(0, 1).flip(0)
        else:
            _, eigenvals, eigenvecs = torch.svd(t, compute_uv=True) # pylint: disable=no-member
            eigenvecs = eigenvecs.transpose(0, 1)
    return eigenvals, eigenvecs

def eigenthings_tensor_utils_batch(t, device=None, out_device='cpu', symmetric=False, topn=-1):
    t = t.to(device)
    assert len(t.shape) == 3
    vals, vecs = [], []
    for t_sample in t:
        eigenvals, eigenvecs = eigenthings_tensor_utils(t_sample, device=device, out_device=out_device, symmetric=symmetric, topn=topn)
        vals.append(eigenvals.unsqueeze(0))
        vecs.append(eigenvecs.unsqueeze(0))
    vals = torch.cat(vals) # pylint: disable=no-member
    vecs = torch.cat(vecs) # pylint: disable=no-member
    return vals, vecs