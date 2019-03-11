# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spsp



def normal_transition_function(x_star, x):
    """
    the transition function
    """
    mu = x
    sigma = 1
    return 1 / (np.pi * 2) ** 0.5 / sigma * np.exp(
        -(x_star - mu) ** 2 / 2 / sigma ** 2)


def mcmc_sampling(tgt_function, n_sample=10000):
    """
    Metroplois_Hastings sampling
    """
    # number of burn-in example
    n_burn_in = 1000

    # the transition function
    p = tgt_function
    q = normal_transition_function

    results = []
    x = 0.1

    while len(results) <= n_sample + n_burn_in:
        u = np.random.rand()
        x_star = np.random.normal(x, 1)
        alpha = min(1, p(x_star) * q(x, x_star) / p(x) / q(x_star, x))
        if u < alpha:
            # accept the new sample
            x = x_star
        results.append(x)

    return np.asarray(results[n_burn_in:])


def pdf_normal(x, mu=0, std=1):
    return 1 / np.sqrt(2 * np.pi * std * std) * np.exp(
        -(x - mu) * (x - mu) / (2 * std * std))


def pdf_gamma(x, k=2, theta=2):
    return 1 / (spsp.gamma(k) * theta ** k) * x ** (k - 1) * np.exp(-x / theta)


def pdf_chi_square(x, k=8):
    half_k = k / 2
    return (1 / (2 ** half_k * spsp.gamma(half_k)) * x ** (half_k - 1) *
            np.exp(-x / 2))

def pdf_Boltzmann_distribution(x, a=5):
    x2 = x*x
    return (2/np.pi)**0.5 * x2 * np.exp(-x2 * 0.5/a/a)/a**3


def plot_samples():
    fig, axes = plt.subplots(4)

    # standard normal distribution
    samples = mcmc_sampling(pdf_normal)
    x = np.linspace(-10, 10, len(samples))
    y = pdf_normal(x)

    axes[0].plot(x, y, color="green", lw=2)
    axes[0].hist(samples, bins=100, normed=True)

    samples = mcmc_sampling(pdf_gamma)
    x = np.linspace(0, 20, len(samples))
    y = pdf_gamma(x)

    axes[1].plot(x, y, color="green", lw=2)
    axes[1].hist(samples, bins=100, normed=True)

    samples = mcmc_sampling(pdf_chi_square)
    x = np.linspace(0, 20, len(samples))
    y = pdf_chi_square(x)

    axes[2].plot(x, y, color="green", lw=2)
    axes[2].hist(samples, bins=100, normed=True)

    samples = mcmc_sampling(pdf_Boltzmann_distribution)
    x = np.linspace(0, 20, len(samples))
    y = pdf_Boltzmann_distribution(x)

    axes[3].plot(x, y, color="green", lw=2)
    axes[3].hist(samples, bins=100, normed=True)

    plt.show()


if __name__ == '__main__':
    plot_samples()
