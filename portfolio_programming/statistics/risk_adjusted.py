# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import numpy as np
import pandas as pd


def Sharpe(series):
    """
    Sharpe ratio
    note the numpy std() function is the population estimator

    Parameters:
    ---------------
    series: array-like
        return of investment (ROI) series

    Returns:
    ----------
    float
        Sharpe ratio
    """
    s = np.asarray(series)
    try:
        val = s.mean() / s.std()
    except FloatingPointError:
        # set 0 when standard deviation is zero
        val = 0
    return val


def Sortino_full(series, mar=0):
    """
    Sortino ratio, using all periods of the series

    Parameters:
    ---------------
    series : array-like
        return of investment (ROI) series
    mar : float
        minimum acceptable return, usually set to 0

    Returns:
    ----------
    float
        Sortino_full ratio
    """
    s = np.asarray(series)
    mean = s.mean()
    semi_std = np.sqrt(((s * ((s - mar) < 0)) ** 2).mean())
    try:
        val = mean / semi_std
    except FloatingPointError:
        # set 0 when semi-standard deviation is zero
        val = 0
    return val, semi_std


def Sortino_partial(series, mar=0):
    """
    Sortino ratio, using only negative roi periods of the series

    Parameters:
    ---------------
    series : array-like
        return of investment (ROI) series
    mar : float
        minimum acceptable return, usually set to 0

    Returns:
    ----------
    float
        Sortino_partial ratio
    """
    s = np.asarray(series)
    mean = s.mean()
    n_neg_period = ((s - mar) < 0).sum()
    try:
        semi_std = np.sqrt(((s * ((s - mar) < 0)) ** 2).sum() / n_neg_period)
        val = mean / semi_std
    except FloatingPointError:
        # set 0 when semi-standard deviation or negative period is zero
        val, semi_std = 0, 0
    return val, semi_std


def maximum_drawdown(series):
    """
    https://en.wikipedia.org/wiki/Drawdown_(economics)
    the peak may be zero
    e.g.
    s= [0, -0.4, -0.2, 0.2]
    peak = [0, 0, 0, 0.2]
    therefore we don't provide relative percentage of mdd

    Parameters:
    ---------------
    series : array-like
        return of investment (ROI) series

    Returns:
    ---------
    float
        maximum dropdown in the series
    """
    s = np.asarray(series)
    peak = pd.expanding_max(s)

    # absolute drawdown
    ad = np.maximum(peak - s, 0)
    mad = np.max(ad)

    return mad
