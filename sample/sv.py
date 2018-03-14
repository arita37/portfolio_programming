# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>

http://docs.pymc.io/notebooks/stochastic_volatility.html
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk

from scipy import optimize

def sv_model():
    n = 400
    returns = np.genfromtxt(pm.get_data("SP500.csv"))
    print(returns)
#    fig, ax = plt.subplots(figsize=(14, 8))
#    ax.plot(returns, label='S&P500')
#    ax.set(xlabel='time', ylabel='returns')
#    ax.legend()

    with pm.Model() as model:
        step_size = pm.Exponential('sigma', 50.)
        s = GaussianRandomWalk('s', sd=step_size,
                               shape=len(returns))

        nu = pm.Exponential('nu', .1)

        r = pm.StudentT('r', nu=nu,
                        lam=pm.math.exp(-2 * s),
                        observed=returns)

        trace = pm.sample(tune=2000, nuts_kwargs=dict(target_accept=.9))
    pm.traceplot(trace, varnames=['sigma', 'nu'])
#    fig, ax = plt.subplots()
#    plt.plot(trace['s'].T, 'b', alpha=.03)
#    ax.set(title=str(s), xlabel='time', ylabel='log volatility')

#   fig, ax = plt.subplots(figsize=(14, 8))
#    ax.plot(returns)
#    ax.plot(np.exp(trace[s].T), 'r', alpha=.03);
#    ax.set(xlabel='time', ylabel='returns')
#    ax.legend(['S&P500', 'stoch vol'])


#    plt.show()

if __name__ == '__main__':
    sv_model()
