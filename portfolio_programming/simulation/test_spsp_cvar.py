# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

"""

import numpy as np
import os
import portfolio_programming as pp
from portfolio_programming.simulation.spsp_cvar import spsp_cvar

def test_spsp_cvar(alpha=0.1):
    symbols = ['test',]
    setting = "compact"
    max_portfolio_size = len(symbols)
    risk_rois = np.zeros(1)
    risk_free_roi = 0
    allocated_risk_wealth = np.array([0., ])
    allocated_risk_free_wealth = 100.
    buy_trans_fee = 0
    sell_trans_fee = 0
    n_scenario = 100
    # predict_risk_rois = np.linspace(0., 1., n_scenario).reshape((1, n_scenario))
    # experiment 1
    # predict_risk_rois = (np.arange(1,101) / n_scenario).reshape(
    #         (1, n_scenario))

    # experiment 2
    # predict_risk_rois = (np.arange(-49, 51)/n_scenario).reshape(
    #     (1, n_scenario))

    # experiment 3
    predict_risk_rois = (np.arange(-24, 76) / n_scenario).reshape(
        (1, n_scenario))

    predict_risk_free_roi = 0


    res = spsp_cvar(symbols, setting, max_portfolio_size,
                risk_rois, risk_free_roi, allocated_risk_wealth,
                allocated_risk_free_wealth, buy_trans_fee, sell_trans_fee,
                alpha, predict_risk_rois, predict_risk_free_roi, n_scenario)
    print("alpha=",alpha)
    print(res)
    return res

def cvar_alpha_plot():
    import matplotlib.pyplot as plt

    xs =  np.arange(1, 100)
    alphas = xs/100.
    cvars = []
    vars = []
    buys = []
    sells = []
    for alpha in alphas:
        res = test_spsp_cvar(alpha)
        cvars.append(res['CVaR'])
        vars.append(res['VaR'])
        buys.append(float(res['amounts'].loc['test', 'buy']))
        sells.append(float(res['amounts'].loc['test', 'sell']))

    fig, ax = plt.subplots(4)
    fig.suptitle('Experiment 3', fontsize=24)
    var_ax, cvar_ax, buy_ax, sell_ax = ax[1], ax[0], ax[2], ax[3]

    var_ax.plot(xs, vars)
    var_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    var_ax.set_ylabel('VaR', fontsize=20)
    var_ax.set_xlim(1, 100)
    var_ax.grid(True)

    cvar_ax.plot(xs, cvars)
    cvar_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    cvar_ax.set_ylabel('CVaR', fontsize=20)
    cvar_ax.set_xlim(1, 100)
    cvar_ax.grid(True)

    buy_ax.plot(xs, buys)
    buy_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    buy_ax.set_ylabel('buy amount', fontsize=20)
    buy_ax.set_xlim(1, 100)
    buy_ax.grid(True)

    sell_ax.plot(xs, sells)
    sell_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    sell_ax.set_ylabel('sell amount', fontsize=20)
    sell_ax.set_xlim(1, 100)
    sell_ax.grid(True)

    fig.set_size_inches(16, 9)
    fig_path = os.path.join(pp.TMP_DIR, 'cvar_test.png')
    plt.savefig(fig_path, dpi=240, format='png')
    plt.show()

if __name__ == '__main__':
    # test_spsp_cvar()
    cvar_alpha_plot()