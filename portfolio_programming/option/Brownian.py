# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Brownian_process(dev=2, n_rv=100, n_step=1000):
    points = np.random.randn(n_rv, n_step) * dev
    ys = points.cumsum(axis=1)
    ys = ys - ys[:, 0][:, np.newaxis]
    df = pd.DataFrame(ys.T)
    df.plot(title='Brownian process ($\mu$=0, $\sigma^2$={:.2f}): {}-{}'.format(
        dev*dev, n_rv, n_step),
        legend=False)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title('points mean')
    ax.plot(points.mean(axis=0))

    fig2 = plt.figure()
    ax = fig2.gca()
    ax.set_title('ys mean')
    ax.plot(ys.mean(axis=0))

    fig3 = plt.figure()
    ax = fig3.gca()
    ax.set_title('deviation')
    ax.plot(ys.std(axis=0))

    plt.show()



def main():
    Brownian_process()

if __name__ == '__main__':
    main()
