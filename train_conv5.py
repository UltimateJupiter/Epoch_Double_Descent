from data.linear import get_linear_data
from models.utils import init_network
from models import *
from data import *

import torch
import random
import numpy as np

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default configs

feat_dim = 25
out_dim = 1

hidden_dim = 250
n_samples = 100
n_epoch = 100
seed = 0

train_dl, test_dl, _ = get_cifar_10(batch_size=32)

if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

verbose_freq = 2

net = Conv4FC1()
net = net.to(device)

n_layers = len(net.layers)

print("# layers: {}".format(len(net.layers)))

# Naive setting
scales = np.array([1] * n_layers)
lr = np.array([0.01] * n_layers)

loss_fn = torch.nn.CrossEntropyLoss()
risk_fn = loss_fn
# loss_fn = risk_fn

def train():

    # train the network
    losses = []
    risks = []

    risks_x = []
    assert len(lr) == len(net.layers)

    steps = 0

    init_network(net, scales=scales)
    for e in range(n_epoch):
        
        for (X_train, y_train) in iter(train_dl):
            
            y_pred = net(X_train)
            loss = loss_fn(y_pred, y_train)
            losses.append(loss.item())

            if steps % verbose_freq == 0:
                print(steps, loss.item())

            net.zero_grad()
            loss.backward()
            steps += 1

            with torch.no_grad():
                for i, layer in enumerate(net.layers):
                    for name, param in layer.named_parameters():
                        param.data -= lr[i] * param.grad
                
        with torch.no_grad():
            total_risk = []
            for (X_test, y_test) in iter(test_dl):
                yt_pred = net(X_test)
                risk = risk_fn(yt_pred, y_test)
                total_risk.append(risk.item())
            risks.append(np.mean(total_risk))
            risks_x.append(steps)

        
    
    info = []
    return losses, [risks, risks_x], info

def main():

    # Modify setting here
    losses, [risks, risks_x], info = train()
    plt.plot(np.arange(len(risks)), risks)
    plt.plot(np.arange(len(risks)), risks)
    plt.ylim([min(risks) - 1, max(risks) + 1])
    # plt.xlim([1, 1000])
    plt.xscale('log')
    
    pic_name = net.name + '_tmp' # modify the name for grid searc
    plt.savefig("{}.jpg".format(pic_name))

main()