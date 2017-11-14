# -*- coding: utf-8 -*-
"""

Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

bootstrap methods
"""

import numpy as np


def bootstrap(series):
    """
    simple bootstrap, sample with replacement

    Parameters:
    ---------------
    series : array-like
        data series with length n_period

    Returns:
    ---------------
    array-like
        samples has the same length as input data
    """
    n_period = len(series)
    s_indices = np.random.randint(0, n_period, n_period)
    samples = series[s_indices]

    return samples


def stationary_bootstrap(series, q_value=0.5):
    """
    Dimitris N. Politis and Joseph P. Romano, "The stationary bootstrap,"
    Journal of the American Statistical Association, pp. 1303-1313, 1994.

    Parameters:
    ---------------
    series : array-like
        data series with length n_period
    q_value : float
        a parameter for controlling the sample block size.
        if q_value = 0.5, it means block size = 1/Q = 2.

    Returns:
    ---------------
    array-like
        samples has the same length as input data

    """
    n_period = len(series)

    s_indices = np.zeros(n_period, dtype=np.int)
    s_indices[0] = np.random.randint(0, n_period)

    for t in range(1, n_period):
        u = np.random.rand()
        if u < q_value:
            s_indices[t] = np.random.randint(0, n_period)
        else:
            s_indices[t] = s_indices[t - 1] + 1
            if s_indices[t] >= n_period:
                s_indices[t] = 0

    samples = series[s_indices]

    return samples


if __name__ == '__main__':
    pass
