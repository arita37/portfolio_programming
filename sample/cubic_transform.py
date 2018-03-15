# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import scipy.special as spsp
import scipy.stats as spstats


def cubic_function(cubic_params, sample_moments, tgt_moments):
    """
    Parameters:
    ----------------
    cubic_params: (a,b,c,d), four floats
    sample_moments: numpy.array, shape:(12,), 1~12 moments of samples
    tgt_moments: numpy.array, shape:(4,), 1~4th moments of target
    """

    a, b, c, d = cubic_params
    ex = sample_moments
    ey = tgt_moments

    a2 = a * a
    a3 = a2 * a
    a4 = a2 * a2

    b2 = b * b
    b3 = b2 * b
    b4 = b2 * b2

    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2

    d2 = d * d
    d3 = d2 * d
    d4 = d2 * d2

    ab = a * b
    ac = a * c
    ad = a * d
    acd = ac * d
    bd = b * d
    bc = b * c
    bcd = bc * d
    cd = c * d

    v1 = (a + b * ex[0] + c * ex[1] + d * ex[2] - ey[0])

    v2 = (d2 * ex[5] +
          2 * cd * ex[4] +
          (2 * bd + c2) * ex[3] +
          2 * (ad + bc) * ex[2] +
          (2 * ac + b2) * ex[1] +
          2 * ab * ex[0] +
          a2 - ey[1])

    v3 = (d3 * ex[8] +
          (3 * c * d2) * ex[7] +
          3 * (b * d2 + c2 * d) * ex[6] +
          (3 * a * d2 + 6 * bcd + c3) * ex[5] +
          3 * (2 * acd + b2 * d + b * c2) * ex[4] +
          (a * (6 * bd + 3 * c2) + 3 * b2 * c) * ex[3] +
          (3 * a2 * d + 6 * a * bc + b3) * ex[2] +
          3 * (a2 * c + a * b2) * ex[1] +
          3 * a2 * b * ex[0] +
          a3 - ey[2])

    v4 = (d4 * ex[11] +
          (4 * cd * d2) * ex[10] +
          (4 * bd * d2 + 6 * c2 * d2) * ex[9] +
          4 * (ad * d2 + 3 * bc * d2 + c3 * d) * ex[8] +
          (12 * ac * d2 + 6 * b2 * d2 + 12 * bd * c2 + c4) * ex[7] +
          4 * (3 * ad * (bd + c2) + bc * (3 * bd + c2)) * ex[6] +
          (6 * a2 * d2 + ac * (24 * bd + 4 * c2) +
           4 * b3 * d + 6 * b2 * c2) * ex[5] +
          (12 * a2 * cd + 12 * ab * (bd + c2) + 4 * b2 * bc) * ex[4] +
          (a2 * (12 * bd + 6 * c2) + 12 * ac * b2 + b4) * ex[3] +
          4 * a * (a * ad + 3 * a * bc + b3) * ex[2] +
          a2 * (4 * ac + 6 * b2) * ex[1] +
          (4 * a2 * ab) * ex[0] +
          a4 - ey[3])

    return v1, v2, v3, v4


def cubic_transform_sampling(tgt_moments, n_sample=10000):
    # define infinity
    INFINITY = 1e10

    # to generate samples Y with zero mean, and unit variance
    ns = float(n_sample)
    ns_m1 = ns - 1.
    ns_m1_2 = ns_m1 * ns_m1
    ns_m2 = ns - 2.
    ns_m3 = ns - 3.
    ns2 = ns * ns

    y_moments = np.zeros(4)
    y_moments[1] = ns_m1 / ns
    y_moments[2] = (tgt_moments[2] * ns_m1 * ns_m2 / ns2)
    y_moments[3] = ((tgt_moments[3] + 3 * ns_m1_2 / ns_m2 /
                     ns_m3) * ns_m2 * ns_m3 * ns_m1_2 / (ns2 - 1) / ns2)

    results = np.zeros(n_sample)
    # iteration for find good start samples
    max_start_iter = 5

    # cubic transform iteration
    max_cubic_iter = 5

    # error
    max_cubic_err = 1e-5

    cubic_err, best_cub_err = INFINITY, INFINITY
    for _ in range(max_start_iter):
        # each random variable consists of n_scenario random sample
        tmp_out = np.random.randn(n_sample)

        # loop until cubic transform converge
        for cub_iter in range(max_cubic_iter):

            # 1~12th moments of the random samples
            ex = np.asarray([(tmp_out ** (idx + 1)).mean()
                             for idx in range(12)])

            # find corresponding cubic parameters
            x_init = np.array([0., 1., 0., 0.])
            out = spopt.leastsq(cubic_function, x_init, args=(ex, y_moments),
                                full_output=True, ftol=1E-12,
                                xtol=1E-12)
            cubic_params = out[0]
            cubic_err = np.sum(out[2]['fvec'] ** 2)

            # update random samples
            tmp_out = (cubic_params[0] +
                       cubic_params[1] * tmp_out +
                       cubic_params[2] * (tmp_out ** 2) +
                       cubic_params[3] * (tmp_out ** 3))

            if cubic_err < max_cubic_err:
                # break starter loop
                break
            else:
                print("cub_iter:{}, cub_error: {}, not converge".format(
                    cub_iter, cubic_err))

        # accept current samples
        if cubic_err < best_cub_err:
            best_cub_err = cubic_err
            results = tmp_out

    # rescale data to original moments
    results = results * tgt_moments[1] + tgt_moments[0]

    return results


#
def statistics(samples):
    return "{:.2f} {:.2f} {:.2f} {:.2f}".format(
        samples.mean(),
        samples.std(ddof=1),
        spstats.skew(samples, bias=False),
        spstats.kurtosis(samples, bias=False))


def plot_samples():
    fig, axes = plt.subplots(4)

    mu, std, skew, ex_kurt = 10, 3, 0, 0
    samples = cubic_transform_sampling([mu, std, skew, ex_kurt])
    axes[0].set_title('normal')
    x = np.linspace(-20, 20, len(samples))
    y = 1 / np.sqrt(2 * np.pi * std * std) * np.exp(
        -(x - mu) * (x - mu) / (2 * std * std))

    axes[0].plot(x, y, color="green", lw=2)
    axes[0].hist(samples, bins=100, normed=True)
    print(statistics(samples))

    # student(nu)
    nu = 8.
    mu, std, skew, ex_kurt = 0, np.sqrt(nu / (nu - 2)), 0, 6 / (nu - 4)
    samples = cubic_transform_sampling([mu, std, skew, ex_kurt])
    axes[1].set_title('student(nu={})'.format(nu))
    x = np.linspace(-20, 20, len(samples))
    y = spsp.gamma((nu + 1) / 2) / np.sqrt(nu * np.pi) / spsp.gamma(nu / 2) * (
            1 + x * x / nu) ** (-(nu + 1) / 2)

    axes[1].plot(x, y, color="green", lw=2)
    axes[1].hist(samples, bins=100, normed=True)
    print(statistics(samples))

    # chi-square(k)
    chi_k = 8.
    half_k = chi_k / 2
    samples = cubic_transform_sampling([chi_k, np.sqrt(2 * chi_k), np.sqrt(
        8 / chi_k), 12 / chi_k])
    axes[2].set_title('chi_square(k={}))'.format(chi_k))
    x = np.linspace(0, 20, len(samples))
    y = 1 / (2 ** (half_k) * spsp.gamma(half_k)) * x ** (half_k - 1) * np.exp(
        -x / 2)

    axes[2].plot(x, y, color="green", lw=2)
    axes[2].hist(samples, bins=100, normed=True)
    print(statistics(samples))

    # gamma(k, theta)
    k = 10
    theta = 1
    samples = cubic_transform_sampling([k * theta,
                                        np.sqrt(k * theta * theta),
                                        2 / np.sqrt(k),
                                        6. / k
                                        ])
    axes[3].set_title('Gamma(k,theta)=({}, {:.2f})'.format(k, theta))
    x = np.linspace(0, 20, len(samples))
    y = 1 / (spsp.gamma(k) * theta ** k) * x ** (k - 1) * np.exp(-x / theta)

    axes[3].plot(x, y, color="green", lw=2)
    axes[3].hist(samples, bins=100, normed=True)
    print(statistics(samples))

    plt.show()


if __name__ == '__main__':
    plot_samples()
