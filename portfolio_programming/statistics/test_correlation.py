# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

"""

import numpy as np
import scipy.stats as sp_stats
from portfolio_programming.statistics.correlation import (
    Pearson_rho, Spearman_rho, Kendall_tau)


def test_Pearson_rho(n_series=100, n_point=100):
    """
    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.correlate.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    """
    mtx1 = np.random.randn(n_series, n_point)
    mtx2 = np.random.randn(n_series, n_point)

    for rdx in range(n_series):
        s1, s2 = mtx1[rdx, :], mtx2[rdx, :]
        np_val = np.corrcoef(s1, s2)[0, 1]
        my_val = Pearson_rho(s1, s2)
        sp_val = sp_stats.pearsonr(s1, s2)[0]

        np.testing.assert_almost_equal(np_val, my_val)
        np.testing.assert_almost_equal(sp_val, my_val)


def test_Spearman_rho(n_series=100, n_point=100):
    """
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats
    .spearmanr.html#scipy.stats.spearmanr
    """
    mtx1 = np.random.randn(n_series, n_point)
    mtx2 = np.random.randn(n_series, n_point)
    for rdx in range(n_series):
        s1, s2 = mtx1[rdx, :], mtx2[rdx, :]
        my_val = Spearman_rho(s1, s2)
        sp_val = sp_stats.spearmanr(s1, s2)[0]

        np.testing.assert_almost_equal(sp_val, my_val)


def test_Kendall_tau(n_series=1000, n_point=20):
    """
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kendalltau.html#scipy.stats.kendalltau
    """
    mtx1 = np.random.randn(n_series, n_point)
    mtx2 = np.random.randn(n_series, n_point)
    for rdx in range(n_series):
        s1, s2 = mtx1[rdx, :], mtx2[rdx, :]
        my_val = Kendall_tau(s1, s2)
        sp_val = sp_stats.kendalltau(s1, s2)[0]

        np.testing.assert_almost_equal(sp_val, my_val)


if __name__ == '__main__':
    test_Pearson_rho()
    test_Spearman_rho()
    test_Kendall_tau()
