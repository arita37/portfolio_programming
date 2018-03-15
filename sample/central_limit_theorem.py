# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import matplotlib.pyplot as plt
import numpy as np


def get_normal(n_sample, n):
    xsum = []
    for i in range(n_sample):
        # 利用中心极限定理，[0,1)均匀分布期望为0.5，方差为1/12
        tsum = (np.mean(np.random.uniform(0, 1, n)) - 0.5) * np.sqrt(12 * n)
        xsum.append(tsum)
    return xsum


def central_limit_theorem():
    # 生成10000个数，观察它们的分布情况
    n_sample = 10000
    # 观察n选不同值时，对最终结果的影响
    N = [1, 2, 10, 1000]

    plt.figure(figsize=(10, 20))
    subi = 220
    for index, n in enumerate(N):
        subi += 1
        plt.subplot(subi)
        normalsum = get_normal(n_sample, n)
        # 绘制直方图
        plt.hist(normalsum, np.linspace(-4, 4, 80), facecolor="green",
                 label="n={0}".format(n))
        plt.ylim([0, 450])
        plt.legend()

    plt.show()

if __name__ == '__main__':
    central_limit_theorem()
