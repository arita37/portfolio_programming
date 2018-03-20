# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

The SPSP CVaR experiments are not able to run in parallel setting(ipyparallel)
because ofthe complex setting of pyomo.
"""

import glob
import json
import logging
import os
import sys
import datetime as dt

import numpy as np
import xarray as xr

import portfolio_programming as pp
import portfolio_programming.simulation.spsp_cvar


def run_SPSP_CVaR(setting, scenario_set_idx, exp_start_date, exp_end_date,
                  max_portfolio_size, rolling_window_size, alpha, n_scenario):
    risky_roi_xarr = xr.open_dataarray(
        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)

    candidate_symbols = json.load(
        open(pp.TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON))

    if setting == 'compact':
        candidate_symbols = candidate_symbols[:max_portfolio_size]

    n_symbol = len(candidate_symbols)
    risky_rois = risky_roi_xarr.loc[exp_start_date:exp_end_date,
                 candidate_symbols, 'simple_roi']

    exp_trans_dates = risky_rois.get_index('trans_date')
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_symbol),
                                       dims=('symbol',),
                                       coords=(candidate_symbols,))
    initial_risk_free_wealth = 1e6
    print(setting, exp_start_date, exp_end_date, max_portfolio_size,
          rolling_window_size, alpha, n_scenario)
    instance = portfolio_programming.simulation.spsp_cvar.SPSP_CVaR(
        candidate_symbols,
        setting,
        max_portfolio_size,
        risky_rois,
        risk_free_rois,
        initial_risk_wealth,
        initial_risk_free_wealth,
        start_date=exp_trans_dates[0],
        end_date = exp_trans_dates[-1],
        rolling_window_size=rolling_window_size,
        alpha=alpha,
        n_scenario=n_scenario,
        scenario_set_idx=scenario_set_idx,
        print_interval=10
    )
    instance.run()


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(filename)15s %(levelname)10s %(asctime)s\n'
               '%(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", type=str,
                        choices=("compact", "general"),
                        help="SPSP setting")

    parser.add_argument("--year", type=int,
                        choices=range(2005, 2017+1),
                        help="yearly experiments")

    parser.add_argument("-M", "--max_portfolio_size", type=int,
                        choices=range(5, 55, 5),
                        help="max_portfolio_size")

    parser.add_argument("-w", "--rolling_window_size", type=int,
                        choices=range(50, 250, 10),
                        help="rolling window size for estimating statistics.")

    parser.add_argument("-a", "--alpha", type=str,
                        choices=["{:.2f}".format(v / 100.)
                                 for v in range(50, 100, 5)],
                        help="confidence level of CVaR")

    parser.add_argument("--scenario_set_idx", type=int,
                        choices=range(1, 4),
                        default=1,
                        help="pre-generated scenario set index.")
    args = parser.parse_args()

    print("run_SPSP_CVaR in single mode")
    if not args.year:
        run_SPSP_CVaR(args.setting,
                      args.scenario_set_idx,
                      '20050103', '20141231',
                      args.max_portfolio_size,
                      args.rolling_window_size,
                      float(args.alpha),
                      200)
    else:
        years = {
            2005: (dt.date(2005, 1, 3), dt.date(2005, 12, 30)),
            2006: (dt.date(2006, 1, 2), dt.date(2006, 12, 29)),
            2007: (dt.date(2007, 1, 2), dt.date(2007, 12, 31)),
            2008: (dt.date(2008, 1, 2), dt.date(2008, 12, 31)),
            2009: (dt.date(2009, 1, 5), dt.date(2009, 12, 31)),
            2010: (dt.date(2010, 1, 4), dt.date(2010, 12, 31)),
            2011: (dt.date(2011, 1, 3), dt.date(2011, 12, 30)),
            2012: (dt.date(2012, 1, 2), dt.date(2012, 12, 28)),
            2013: (dt.date(2013, 1, 2), dt.date(2013, 12, 31)),
            2014: (dt.date(2014, 1, 2), dt.date(2014, 12, 31)),
            2015: (dt.date(2015, 1, 5), dt.date(2015, 12, 31)),
            2016: (dt.date(2016, 1, 4), dt.date(2016, 12, 30)),
            2017: (dt.date(2017, 1, 3), dt.date(2017, 12, 29))
         }

        run_SPSP_CVaR(args.setting,
                      args.scenario_set_idx,
                      years[args.year][0],
                      years[args.year][1],
                      args.max_portfolio_size,
                      args.rolling_window_size,
                      float(args.alpha),
                      200)
