# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import json
import logging

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.simulation.spsp_cvar import SPSP_CVaR


def run_compact_SPSP_CVaR(n_symbol, rolling_window_size, alpha):
    risky_roi_xarr = xr.open_dataarray(
        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)
    candidate_symbols = json.load(
        open(pp.TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON))[:n_symbol]

    risky_rois = risky_roi_xarr.loc[pp.EXP_START_DATE:pp.EXP_END_DATE,
                 candidate_symbols, 'simple_roi']

    exp_trans_dates = risky_rois.get_index('trans_date')
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_symbol),
                                       dims=('symbol',),
                                       coords=(candidate_symbols,))
    initial_risk_free_wealth = 1e6

    instance = SPSP_CVaR(candidate_symbols,
                         "compact",
                         n_symbol,
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


def run_general_SPSP_CVaR(max_portfolio_size, rolling_window_size, alpha):
    risky_roi_xarr = xr.open_dataarray(
        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)
    candidate_symbols = json.load(
        open(pp.TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON))

    risky_rois = risky_roi_xarr.loc[pp.EXP_START_DATE:pp.EXP_END_DATE,
                 candidate_symbols, 'simple_roi']

    exp_trans_dates = risky_rois.get_index('trans_date')
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_symbol),
                                       dims=('symbol',),
                                       coords=(candidate_symbols,))
    initial_risk_free_wealth = 1e6

    instance = SPSP_CVaR(candidate_symbols,
                         "general",
                         max_portfolio_size,
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
                        level=logging.INFO)
    # run_compact_SPSP_CVaR(10, 90, 0.5)
    run_general_SPSP_CVaR(5, 200, 0.5)
