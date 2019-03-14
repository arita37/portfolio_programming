# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import numpy as np
import scipy.spatial.distance as spdist
import scipy.stats as spstats


def distance_correlation(s1, s2):
    """
    http://en.wikipedia.org/wiki/Distance_correlation
    http://cran.r-project.org/web/packages/energy/energy.pdf

    Parameters:
    --------------
    the same results as R package energy: dcor
    s1: numpy.array
    s2: numpy.array

    if dimension of S1 or S2 >=2,
    then row: data, column: feature

    Returns:
    ---------
    0<= coefficient <= 1
    """
    s1, s2 = np.asarray(s1), np.asarray(s2)

    if not s1.shape[:-1]:
        # 1D array
        s1 = s1[:, np.newaxis]

    if not s2.shape[:-1]:
        # 1D array
        s2 = s2[:, np.newaxis]

    assert s1.shape[0] == s2.shape[0]

    # pairwise distance
    a = spdist.squareform(spdist.pdist(s1))
    a_row_mean = a.mean(axis=0)
    a_mean = a.mean()

    b = spdist.squareform(spdist.pdist(s2))
    b_row_mean = b.mean(axis=0)
    b_mean = b.mean()

    # double centered distance
    dc_a = a - a_row_mean - a_row_mean[:, np.newaxis] + a_mean
    dc_b = b - b_row_mean - b_row_mean[:, np.newaxis] + b_mean

    # squared sample distance covariance and variances
    sq_dcov = (dc_a * dc_b).mean()
    sq_dvar1 = (dc_a * dc_a).mean()
    sq_dvar2 = (dc_b * dc_b).mean()
    if sq_dvar1 == 0 or sq_dvar2 == 0:
        dcor = 0
        dcov, dvar1, dvar2 = np.sqrt((sq_dcov, sq_dvar1, sq_dvar2))
    else:
        sq_dcor = sq_dcov / np.sqrt(sq_dvar1 * sq_dvar2)
        dcor, dcov, dvar1, dvar2 = np.sqrt(
            (sq_dcor, sq_dcov, sq_dvar1, sq_dvar2))

    return {"dcor": dcor, "dcov": dcov, "dvar1": dvar1, "dvar2": dvar2}



def dcor_test(s1, s2):
    """
    http://en.wikipedia.org/wiki/Distance_correlation
    http://cran.r-project.org/web/packages/energy/energy.pdf

    Parameters:
    --------------
    the same results as R package energy: dcor
    s1: numpy.array
    s2: numpy.array

    if dimension of S1 or S2 >=2,
    then row: data, column: feature

     Returns:
    ---------
    coefficient
    """
    s1, s2 = np.asarray(s1), np.asarray(s2)

    if s1.ndim == 1:
        # 1D array
        s1 = s1[:, np.newaxis]

    if s2.ndim == 1:
        # 1D array
        s2 = s2[:, np.newaxis]

    assert s1.shape[0] == s2.shape[0]
    n_samples = s1.shape[0]

    # pairwise distance (a_col_mean == a_row_mean)
    a = spdist.squareform(spdist.pdist(s1))
    a_row_mean = a.mean(axis=0)
    a_mean = a.mean()

    b = spdist.squareform(spdist.pdist(s2))
    b_row_mean = b.mean(axis=0)
    b_mean = b.mean()

    # double centered distance
    A = a - a_row_mean - a_row_mean[:, np.newaxis] + a_mean
    B = b - b_row_mean - b_row_mean[:, np.newaxis] + b_mean

    # squared sample distance covariance and variances
    sq_dcov, sq_dvar1, sq_dvar2 = np.mean(A * B), np.mean(A * A), np.mean(B * B)
    if sq_dvar1 == 0 or sq_dvar2 == 0:
        dcor = 0
    else:
        sq_dcor = sq_dcov / np.sqrt(sq_dvar1 * sq_dvar2)
        dcor = np.sqrt(sq_dcor)

    # modified A, B
    if n_samples >= 4:
        c = n_samples / (n_samples - 1.)

        a_star = c * (A - a / n_samples)
        a_star.ravel()[::n_samples + 1] = c * (a_row_mean - a_mean)
        a_diag = a_star.ravel()[::n_samples + 1]

        b_star = c * (B - b / n_samples)
        b_star.ravel()[::n_samples + 1] = c * (b_row_mean - b_mean)
        b_diag = b_star.ravel()[::n_samples + 1]

        c2 = n_samples / (n_samples - 2.)
        c3 = 1. / (n_samples * (n_samples - 3.))
        v_xy = c3 * ((a_star * b_star).sum() - c2 * (a_diag * b_diag).sum())
        v_xx = c3 * ((a_star * a_star).sum() - c2 * (a_diag * a_diag).sum())
        v_yy = c3 * ((b_star * b_star).sum() - c2 * (b_diag * b_diag).sum())

        v_xxyy = v_xx * v_yy
        if v_xxyy > 0:
            m_dcor = v_xy / np.sqrt(v_xxyy)
        else:
            m_dcor = 0

        df = n_samples * (n_samples - 3) / 2 - 1
        try:
            t_stats = np.sqrt(df) * m_dcor / np.sqrt(1. - m_dcor * m_dcor)
            pvalue = 1. - spstats.t.cdf(t_stats, df=df)
        except FloatingPointError:
            # if m_dcor == 1 or -1
            t_stats = None
            pvalue = 0.

    else:
        m_dcor, df, t_stats, pvalue = None, 0, None, None

    return {"dcor": dcor, "modified_dcor": m_dcor,
            "pvalue": pvalue, "t_stats": t_stats, "df": df}
