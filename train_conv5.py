from data.linear import get_linear_data
from models.utils import init_network
from models import *
from data import *
from measure import *

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
n_epoch = 5
seed = 0

if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

train_dl, test_dl, _ = get_cifar_10(batch_size=32, ondev=True)

test_record_freq = 25
train_record_freq = 1
verbose_freq = 1

net = Conv4FC1().to(device)

n_layers = len(net.layers)

print("Model: {} #layers: {}".format(net.name, len(net.layers)))

# Naive setting
scales = np.array([1] * n_layers)
lr = np.array([0.01] * n_layers)

loss_fn = torch.nn.CrossEntropyLoss()
risk_fn = err_onehot_int
# loss_fn = risk_fn

def test(dl, model):
    batch_losses = []
    batch_risks = []
    with torch.no_grad():
        for (X_test, y_test) in iter(dl):
            yt_pred = model(X_test)
            batch_risks.append(risk_fn(yt_pred, y_test).item())
            batch_losses.append(loss_fn(yt_pred, y_test).item())
    return [np.mean(batch_losses), np.mean(batch_risks)]

def train():

    # train the network
    train_x = []
    train_losses = []
    test_losses = []

    test_x = []
    test_risks = []
    train_risks = []

    assert len(lr) == len(net.layers)

    steps = 0

    init_network(net, scales=scales)
    
    for e in range(n_epoch):
        
        for (X_train, y_train) in iter(train_dl):
            
            y_pred = net(X_train)
            loss = loss_fn(y_pred, y_train)

            if steps % train_record_freq == 0:
                risk = risk_fn(y_pred, y_train)
                train_losses.append(loss.item())
                train_risks.append(risk.item())
                train_x.append(steps)

            net.zero_grad()
            loss.backward()
            steps += 1

            with torch.no_grad():
                for i, layer in enumerate(net.layers):
                    for name, param in layer.named_parameters():
                        param.data -= lr[i] * param.grad
            
            if steps % test_record_freq == 0:
                test_loss, test_risk = test(test_dl, net)
                test_risks.append(test_risk)
                test_losses.append(test_loss)
                test_x.append(steps)

        if e % verbose_freq == 0:
            verbose_arg = [e, train_losses[-1], train_risks[-1], test_losses[-1], test_risks[-1]]
            print("Epoch{} | TrainLoss {:.4g} | TrainRisk {:.4g} | TestLoss {:.4g} | TestRisk {:.4g}".format(*verbose_arg))
    
    info = []
    return [train_losses, train_risks, train_x], [test_losses, test_risks, test_x], info

def main():

    # Modify setting here
    [train_losses, train_risks, train_x], [test_losses, test_risks, test_x], info = train()
    
    plt.subplot(121)
    plt.title("loss")
    plt.plot(train_x, train_losses, label='train')
    plt.plot(test_x, test_risks, label='test')
    plt.legend()
    plt.ylim([min(test_losses) - 1, max(test_losses) + 1])
    plt.xscale('log')

    plt.subplot(122)
    plt.title("risk")
    plt.plot(train_x, train_risks, label='train')
    plt.plot(test_x, test_risks, label='test')
    plt.legend()
    # plt.ylim([min(test_risks) - 1, max(test_risks) + 1])
    plt.xscale('log')

    pic_name = net.name + '_tmp' # modify the name for grid searc
    plt.savefig("{}.jpg".format(pic_name))

main()