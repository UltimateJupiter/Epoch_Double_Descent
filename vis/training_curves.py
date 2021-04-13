import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

import numpy as np

def seq_moving_average(x, w):
    base = np.convolve(np.ones_like(x), np.ones(w), 'same')
    return np.convolve(np.array(x), np.ones(w), 'same') / base

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

    
    