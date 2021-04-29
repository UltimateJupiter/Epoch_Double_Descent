import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

import numpy as np

def seq_moving_average(x, w):
    base = np.convolve(np.ones_like(x), np.ones(w), 'same')
    return np.convolve(np.array(x), np.ones(w), 'same') / base

def plot_weight_traj(weight_traj, pic_name):
    [weights_diff, train_x] = weight_traj
    plt.figure(figsize=(10,5))
    plt.title("weights")
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
    for i in range(len(weights_diff)):
        plt.scatter(train_x, weights_diff[i], label=layers[i])
    plt.legend()
    plt.xlabel('Steps')

    fig_desc = 'weights'
    plt.savefig("./figs/{}_{}.jpg".format(pic_name, fig_desc))


def plot_weight_traj2(weight_traj, pic_name):
    [weights_diff, train_x] = weight_traj
    plt.figure(figsize=(10,5))
    plt.title("weights")
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
    for i in range(len(weights_diff)):
        plt.scatter(train_x[2:], weights_diff[i][2:], label=layers[i])
    plt.legend()
    plt.xlabel('Steps')

    fig_desc = 'weights2'
    plt.savefig("./figs/{}_{}.jpg".format(pic_name, fig_desc))


def plot_training_traj(train_traj, test_traj, pic_name, moving_average=True, log_scale=False):

    [train_losses, train_risks, train_x], [test_losses, test_risks, test_x] = train_traj, test_traj

    s_train_losses = seq_moving_average(train_losses, max(10, len(train_losses) // 100))
    s_train_risks = seq_moving_average(train_risks, max(10, len(train_risks) // 100))

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title("loss")
    if moving_average:
        plt.plot(train_x, train_losses, label='train', color='blue', alpha=0.3)
        plt.plot(train_x, s_train_losses, color='blue')
    else:
        plt.plot(train_x, train_losses, label='train', color='blue')
    plt.plot(test_x, test_losses, label='test', color='red')
    plt.legend()
    plt.xlabel('Steps')
    # plt.ylim([min(test_losses) - 1, max(test_losses) + 1])
    if log_scale:
        plt.xscale('log')

    plt.subplot(122)
    plt.title("risk")
    if moving_average:
        plt.plot(train_x, train_risks, label='train', color='blue', alpha=0.3)
        plt.plot(train_x, s_train_risks, color='blue')
    else:
        plt.plot(train_x, train_risks, label='train', color='blue')
    plt.plot(test_x, test_risks, label='test', color='red')
    plt.legend()
    plt.xlabel('Steps')
    # plt.ylim([min(test_risks) - 1, max(test_risks) + 1])
    plt.suptitle(pic_name)
    if log_scale:
        plt.xscale('log')
    
    fig_desc = 'log{}'.format(int(log_scale))

    plt.savefig("./figs/{}_{}.jpg".format(pic_name, fig_desc))

    
    