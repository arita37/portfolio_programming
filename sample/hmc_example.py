# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
http://www.datalearner.com/blog/1051484459699809
"""

import numpy as np


def hmc_harmonic_oscillator():
    # time step size
    delta = 0.1

    # leap frog
    L = 70

    # kinetic function
    func_K = lambda p: p*p/2

    # potential function
    func_U = lambda x: 0.5*x*x
    dU = lambda x: x

    # initial condition
    x0 = -4
    p0 = 1

    # HMC with leap frog method
    # first half step for momentum
    p_step = p0 - delta/2 * dU(x0)

    # first full step for position
    x_step = x0 + delta * p_step

    for idx in range(L):
        # update momentum
        p_step = p_step - delta * dU(x_step)

        # update position
        x_step = x_step + delta * p_step
        print("position:{:.4f}, momentum:{:.4f}".format(x_step, p_step))
    # last half step for momentum
    p_step = p_step - delta * dU(x_step)


if __name__ == '__main__':
    hmc_harmonic_oscillator()