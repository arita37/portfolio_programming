# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

眼球中央
"""

import numpy as np


def Markov_transition():
    # init_prob = np.array([0.21, 0.68, 0.11])
    init_prob = np.random.rand(3)
    trans_mtx = np.array([
        [0.65, 0.28, 0.07],
        [0.15, 0.67, 0.18],
        [0.12, 0.36, 0.52]
    ])

    for idx in range(100):
        print(init_prob)
        init_prob = np.dot(trans_mtx.T, init_prob)


if __name__ == '__main__':
    Markov_transition()