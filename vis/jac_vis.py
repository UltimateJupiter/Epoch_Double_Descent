import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

import numpy as np

def plot_jac_svd(D, splitted_norms, slices, layer_names, pic_name):

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    for i, norms in enumerate(splitted_norms):
        plt.scatter(D, norms, label=layer_names[i])
    plt.xlabel(r"$\sigma_i$")
    plt.title(r"$||\cdot||$")
    plt.legend()

    plt.subplot(122)
    for i, norms in enumerate(splitted_norms):
        s, e = slices[i]
        plt.scatter(D, norms / np.sqrt(e-s), label=layer_names[i])
    plt.xlabel(r"$\sigma_i$")
    plt.title(r"$||\cdot||\ normalized$")
    plt.legend()

    plt.suptitle(pic_name)
    
    plt.savefig("./figs/{}.jpg".format(pic_name))

    