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

import numpy as np
import xarray as xr

import portfolio_programming as pp
import portfolio_programming.simulation.spsp_cvar


def _all_SPSP_CVaR_params(setting):
    """
    "report_SPSP_CVaR_{}_scenario-set-idx{}_{}_{}_M{}_Mc{}_h{}_a{:.2f}_s{
    }.pkl".format(
                self.setting,
                self.scenario_set_idx,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d"),
                self.max_portfolio_size,
                self.n_symbol,
                self.rolling_window_size,
                self.alpha,
                self.n_scenario
            )
    """
    REPORT_FORMAT = "repot_SPSP_CVaR_{setting}_scenario-set-idx{sdx}_{" \
                    "exp_start_date}_{exp_end_date}_M{max_portfolio}_Mc{" \
                    "n_candidate_symbol}_h{rolling_window_size" \
                    "}_a{alpha}_s{n_scenario}.pkl"
    if setting not in ('compact', 'general'):
        raise ValueError('Wrong setting: {}'.format(setting))

    # set_indices = (1, 2, 3)
    set_indices = (1,)
    s_date = pp.SCENARIO_START_DATE.strftime("%Y%m%d")
    e_date = pp.SCENARIO_END_DATE.strftime("%Y%m%d")
    max_portfolio_sizes = range(5, 50 + 5, 5)
    window_sizes = range(60, 240 + 10, 10)
    n_scenarios = [200, ]
    alphas = ["{:.2f}".format(v / 100.) for v in range(50, 100, 10)]

    # dict comprehension
    # key: file_name, value: parameters
    if setting == "compact":
        return {
            REPORT_FORMAT.format(
                setting=setting,
                sdx=sdx,
                exp_start_date=s_date,
                exp_end_date=e_date,
                max_portfolio=m,
                n_candidate_symbol=m,
                rolling_window_size=h,
                alpha=a,
                n_scenario=s
            ): (setting, sdx, s_date, e_date, m, h, float(a), s)
            for sdx in set_indices
            for m in max_portfolio_sizes
            for h in window_sizes
            for a in alphas
            for s in n_scenarios
        }

    elif setting == "general":
        return {
            REPORT_FORMAT.format(
                setting=setting,
                sdx=sdx,
                exp_start_date=s_date,
                exp_end_date=e_date,
                max_portfolio=m,
                n_candidate_symbol=50,
                rolling_window_size=h,
                alpha=a,
                n_scenario=s
            ): (setting, sdx, s_date, e_date, m, h, float(a), s)
            for sdx in set_indices
            for m in max_portfolio_sizes
            for h in window_sizes
            for a in alphas
            for s in n_scenarios
        }


def checking_existed_SPSP_CVaR_report(setting, report_dir=None):
    """
    return unfinished experiment parameters.
    """
    if report_dir is None:
        report_dir = pp.REPORT_DIR
    all_reports = _all_SPSP_CVaR_params(setting)

    os.chdir(report_dir)
    existed_reports = glob.glob("*.pkl")
    for report in existed_reports:
        all_reports.pop(report, None)

    # unfinished params
    return all_reports


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
    run_SPSP_CVaR(args.setting,
                  args.scenario_set_idx,
                  '20050103', '20141231',
                  args.max_portfolio_size,
                  args.rolling_window_size,
                  float(args.alpha),
                  200)
