# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import numpy as np


def DTW(S1, S2, window=5):
    """
    dynamic time warping with constraint window size
    the larger distance, the less similar between the series
    
    Parameters:
    --------------
    S1 : array-like
    S2 : array-like
    window: positive integer

    Returns:
    ---------------
    positive integer
        DTW distance
    """
    S1, S2 = np.asarray(S1), np.asarray(S2)
    window = max(window, abs(S1.size - S2.size))
    mtx = np.ones((S1.size + 1, S2.size + 1)) * np.inf
    mtx[0, 0] = 0.

    for idx in range(S1.size):
        low, high = max(0, idx - window), min(S2.size, idx + window + 1)
        for jdx in range(low, high):
            cost = abs(S1[idx] - S2[jdx])
            mtx[idx + 1, jdx + 1] = cost + min(mtx[idx, jdx + 1],
                                               mtx[idx + 1, jdx],
                                               mtx[idx, jdx]  # match
                                               )
    return mtx[idx, jdx]


if __name__ == '__main__':
    pass
