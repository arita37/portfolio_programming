# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import json
import numpy as np
import pandas as pd

import portfolio_programming as pp
from portfolio_programming.simulation.spsp_cvar import SPSP_CVaR


def run_SPSP_CVaR():
    n_stock = 5
    candidate_symbols = json.load(open(pp.TAIEX_SYMBOL_JSON))[:n_stock]
    pnl = pd.read_pickle(pp.TAIEX_PANEL_PKL)
    risky_rois = pnl.loc[pp.EXP_START_DATE:pp.EXP_END_DATE,
                 candidate_symbols, 'simple_roi'].T

    n_exp_dates = len(risky_rois.index)
    risk_free_rois = pd.Series(np.zeros(n_exp_dates), risky_rois.index)
    initial_risk_wealth = np.zeros(n_stock)
    initial_risk_free_wealth = 1e6

    instance = SPSP_CVaR(candidate_symbols,
                         "compact",
                         n_stock,
                         risky_rois,
                         risk_free_rois,
                         initial_risk_wealth,
                         initial_risk_free_wealth,
                         rolling_window_size=240,
                         alpha=0.95,
                         scenario_set_idx=1
                         )
    instance.run()


if __name__ == '__main__':
    run_SPSP_CVaR()
