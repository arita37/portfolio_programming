# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import json
import numpy as np
import pandas as pd
import xarray as xr
import logging

import portfolio_programming as pp
from portfolio_programming.simulation.spsp_cvar import SPSP_CVaR


def run_SPSP_CVaR(n_stock, rolling_window_size, alpha):
    risky_roi_xarr = xr.open_dataarray(pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)
    candidate_symbols = json.load(open(pp.TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON))[:n_stock]

    risky_rois = risky_roi_xarr.loc[pp.EXP_START_DATE:pp.EXP_END_DATE,
                 candidate_symbols, 'simple_roi']

    exp_trans_dates = risky_rois.get_index('trans_date')
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_stock),
                                       dims=('symbol',),
                                       coords=(candidate_symbols,))
    initial_risk_free_wealth = 1e6

    instance = SPSP_CVaR(candidate_symbols,
                         "compact",
                         n_stock,
                         risky_rois,
                         risk_free_rois,
                         initial_risk_wealth,
                         initial_risk_free_wealth,
                         rolling_window_size=rolling_window_size,
                         alpha=alpha,
                         scenario_set_idx=1,
                         print_interval=10
                         )
    instance.run()


if __name__ == '__main__':
    logging.basicConfig(format='%(filename)15s %(levelname)10s %(asctime)s\n'
                               '%(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.DEBUG)
    run_SPSP_CVaR(5, 150, 0.8)
