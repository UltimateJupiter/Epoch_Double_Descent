from data.linear import get_linear_data
from models.utils import init_network
from models import *
from data import *
from measure import *
from vis import *

import torch
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default configs

n_epoch = 100
seed = 0

if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

train_dl, test_dl, _ = None, None, None
# train_dl, test_dl, _ = get_cifar_10(batch_size=32, ondev=True, device=device)

test_record_freq = 32
train_record_freq = 8
verbose_freq = 10

net = VGG11_nobn().to(device)
n_layers = len(net.layers)

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
            X_test, y_test = X_test.to(device), y_test.to(device)
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
    D, splitted_norms, slices, layer_names = get_jacobian_svd(train_dl, net, batchsize=64, num_classes=10, device='cpu')
    plot_jac_svd(D, splitted_norms, slices, layer_names, 'svd_vgg_nobn_64')
    exit()
    
    for e in range(n_epoch):
        
        for (X_train, y_train) in iter(train_dl):
            
            if steps % test_record_freq == 0:
                test_loss, test_risk = test(test_dl, net)
                test_risks.append(test_risk)
                test_losses.append(test_loss)
                test_x.append(steps)

            X_train, y_train = X_train.to(device), y_train.to(device)
            
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

        if e % verbose_freq == 0:
            verbose_arg = [e, steps, train_losses[-1], train_risks[-1], test_losses[-1], test_risks[-1]]
            log("Epoch{} Step{} | TrainLoss {:.4g} | TrainRisk {:.4g} | TestLoss {:.4g} | TestRisk {:.4g}".format(*verbose_arg))
    
    info = []
    return [train_losses, train_risks, train_x], [test_losses, test_risks, test_x], info

def main():

    # Modify setting here
    global lr 
    lr = np.array([0.1] * n_layers)
    lr[-1] = 0.01

    global n_epoch
    n_epoch = 100

    global train_dl, test_dl
    train_dl, test_dl, _ = get_cifar_10(batch_size=128, ondev=True, device=device, noise_level=0.2)

    exp_name = net.name + '_noise0.2_lr0.1_fc0.01_100epoch' # modify the name for grid search
    print(exp_name)

    train_res = train()


    train_traj, test_traj, info = train_res

    log_name = './log/{}.pkl'.format(exp_name)
    torch.save([train_res, exp_name], log_name)

    plot_training_traj(train_traj, test_traj, exp_name)
    plot_training_traj(train_traj, test_traj, exp_name, log_scale=True)

def plot_log():
    exp_name = net.name + '_100epoch' # modify the name for grid search
    
    log_name = './log/{}.pkl'.format(exp_name)
    [train_traj, test_traj, info], exp_name = torch.load(log_name)
    plot_training_traj(train_traj, test_traj, exp_name)

main()
# plot_log()