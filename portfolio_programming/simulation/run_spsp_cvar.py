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


def run_SPSP_CVaR(setting, max_portfolio_size, rolling_window_size, alpha,
                  sceenario_set_idx):
    risky_roi_xarr = xr.open_dataarray(
        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)

    candidate_symbols = json.load(
        open(pp.TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON))

    if setting == 'compact':
        candidate_symbols = candidate_symbols[:max_portfolio_size]

    n_symbol = len(candidate_symbols)
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
                         setting,
                         max_portfolio_size,
                         risky_rois,
                         risk_free_rois,
                         initial_risk_wealth,
                         initial_risk_free_wealth,
                         rolling_window_size=rolling_window_size,
                         alpha=alpha,
                         scenario_set_idx=sceenario_set_idx,
                         print_interval=10
                         )
    instance.run()

def parallel_run_SPSP_CVaR():
    import ipyparallel as ipp
    from time import sleep
    from IPython.display import clear_output

    settings = ("compact", )
    max_portfolio_sizes = range(5, 50 + 5, 5)
    window_sizes = range(120, 240 + 10, 10)
    alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 10)]
    set_indices = (1,)

    params = [
        (setting, m, h, float(alpha), sdx)
        for setting in settings
        for m in max_portfolio_sizes
        for h in window_sizes
        for alpha in alphas
        for sdx in set_indices
    ]

    # task interface
    rc = ipp.Client(profile='ssh')
    dv = rc[:]
    dv.use_dill()

    with dv.sync_imports():
        import sys
        import platform
        import os

    def name_pid():
        return "node:{}, pid:{}".format(platform.node(), os.getpid())

    infos = dv.apply_sync(name_pid)
    for info in infos:
        print(info)

    lbv = rc.load_balanced_view()
    print("start map unfinished parameters to load balance view.")
    ar = lbv.map_async(lambda x: run_SPSP_CVaR(*x), params)

    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        for stdout in ar.stdout[-10:]:
            if stdout:
                print(stdout)
        sys.stdout.flush()
        sleep(2)


if __name__ == '__main__':
    logging.basicConfig(format='%(filename)15s %(levelname)10s %(asctime)s\n'
                               '%(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', 
                        action='store_true',
                        help="parallel mode or not")

    parser.add_argument("--setting", type=str,
                        choices=("compact", "general"),
                        help="SPSP setting")

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

    parser.add_argument("--scenario-set-idx", type=int,
                        choices=range(1, 4),
                        default=1,
                        help="pre-generated scenario set index.")
    args = parser.parse_args()

    if args.parallel:
        print("run_SPSP_CVaR in parallel mode")
        parallel_run_SPSP_CVaR()
    else:
        print("run_SPSP_CVaR in single mode")
        run_SPSP_CVaR(args.setting, args.max_portfolio_size,
                      args.rolling_window_size,
                      float(args.alpha),
                      args.scenario_set_idx)
