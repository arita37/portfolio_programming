# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import numpy as np
import scipy.stats as sp_stats


def Pearson_rho(s1, s2):
    """
    Pearson correlation coefficient
    moment-based correlation

    Parameters:
    --------------
    s1 : array-like
    s2 : array-like

    Returns:
    ---------
    float
        Pearson correlation coefficient, 0<=rho<=1
    """
    if len(s1) != len(s2):
        raise ValueError("length of S1 {} != length of s2 {}.".format(
            len(s1), len(s2)))

    s1, s2 = np.asarray(s1), np.asarray(s2)
    mu_1, mu_2 = s1.mean(), s2.mean()

    centered1 = s1 - mu_1
    centered2 = s2 - mu_2
    cov = (centered1 * centered2).sum()
    var1 = (centered1 * centered1).sum()
    var2 = (centered2 * centered2).sum()
    rho = cov / np.sqrt(var1 * var2)
    return rho


def Spearman_rho(s1, s2):
    """
    rank-based correlation

    Parameters:
    --------------
    s1 : array-like
    s2 : array-like

    Returns:
    ---------
    float
       Spearman correlation coefficient, 0<=rho<=1
    """
    if len(s1) != len(s2):
        raise ValueError("length of S1 {} != length of s2 {}.".format(
            len(s1), len(s2)))

    s1, s2 = np.asarray(s1), np.asarray(s2)

    r1, r2 = sp_stats.rankdata(s1, 'min'), sp_stats.rankdata(s2, 'min')
    mu_1, mu_2 = r1.mean(), r2.mean()
    centered1 = r1 - mu_1
    centered2 = r2 - mu_2
    cov = (centered1 * centered2).sum()
    var1 = (centered1 * centered1).sum()
    var2 = (centered2 * centered2).sum()
    rho = cov / np.sqrt(var1 * var2)
    return rho


def Kendall_tau(s1, s2):
    """
    rank-based correlation, O(n^2) implementation

    Parameters:
    --------------
    s1 : array-like
    s2 : array-like

    Returns:
    ---------
    float
       Kendall correlation coefficient, 0<=tau<=1
    """
    if len(s1) != len(s2):
        raise ValueError("length of S1 {} != length of s2 {}.".format(
            len(s1), len(s2)))

    n_data = len(s1)
    n_pair = n_data * (n_data - 1) / 2

    n_concordant, n_discordant = 0, 0
    for idx in range(n_data):
        for jdx in range(idx + 1, n_data):
            if ((s1[idx] > s1[jdx] and s2[idx] > s2[jdx]) or
                    (s1[idx] < s1[jdx] and s2[idx] < s2[jdx])):
                n_concordant += 1
            elif ((s1[idx] > s1[jdx] and s2[idx] < s2[jdx]) or
                      (s1[idx] < s1[jdx] and s2[idx] > s2[jdx])):
                n_discordant += 1

    tau = float(n_concordant - n_discordant) / n_pair
    return tau
