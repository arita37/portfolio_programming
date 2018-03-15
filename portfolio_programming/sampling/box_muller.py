# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import numpy as np
import matplotlib.pyplot as plt


def box_muller_samples(n_sample=100000):
    u1 = np.random.rand(n_sample)
    u2 = np.random.rand(n_sample)
    x = np.sqrt(-2*np.log(u1))
    y = 2*np.pi*u2
    z1 = x * np.cos(y)
    z2 = x * np.sin(y)

    f, ax = plt.subplots(3, 1)
    ax[0].scatter(z1,z2)
    ax[1].hist(z1, bins=100)
    ax[2].hist(z2, bins=100)
    plt.show()




if __name__ == '__main__':
    box_muller_samples()