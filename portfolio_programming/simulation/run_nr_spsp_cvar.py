# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import logging
import sys

import numpy as np
import xarray as xr

import portfolio_programming as pp
import portfolio_programming.simulation.spsp_cvar


def get_experts(expert_group_name):
    """
    pattern 'h100-150-10_a50-70-5
    """
    h, a = expert_group_name.split('_')
    h_params = h[1:].split('-')
    a_params = a[1:].split('-')
    h_start, h_end, h_inc = map(int, h_params)
    a_start, a_end, a_inc = map(int, a_params)
    experts = [(h, a / 100)
               for h in range(h_start, h_end + h_inc, h_inc)
               for a in range(a_start, a_end + a_inc, a_inc)]
    return experts


def run_NER_SPSP_CVaR(exp_name, nr_strategy, nr_param, expert_group_name,
                      group_name, n_scenario, scenario_set_idx,
                      exp_start_date, exp_end_date):
    market = group_name[:2]
    if market == "TW":
        risky_roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    elif market == "US":
        risky_roi_xarr = xr.open_dataarray(pp.DJIA_2005_NC)

    candidate_symbols = pp.GROUP_SYMBOLS[group_name]
    n_symbol = len(candidate_symbols)
    risky_rois = risky_roi_xarr.loc[exp_start_date:exp_end_date,
                 candidate_symbols, 'simple_roi']
    exp_trans_dates = risky_rois.get_index('trans_date')
    # print('exp_trans_dates:', exp_trans_dates)
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_symbol),
                                       dims=('symbol',),
                                       coords=(candidate_symbols,))
    initial_risk_free_wealth = 100
    print(exp_name, nr_strategy, nr_param, expert_group_name,
          group_name, n_scenario, scenario_set_idx,
          exp_start_date, exp_end_date)

    experts = get_experts(expert_group_name)

    instance = portfolio_programming.simulation.spsp_cvar.NER_SPSP_CVaR(
        nr_strategy,
        nr_param,
        expert_group_name,
        experts,
        group_name,
        candidate_symbols,
        risky_rois,
        risk_free_rois,
        initial_risk_wealth,
        initial_risk_free_wealth,
        start_date=exp_trans_dates[0],
        end_date=exp_trans_dates[-1],
        n_scenario=n_scenario,
        scenario_set_idx=scenario_set_idx,
        print_interval=1
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

    parser.add_argument("--nr_strategy", type=str, help="no-regret strategy")
    parser.add_argument("--nr_param", type=float,
                        help="no-regret strategy parameter")
    parser.add_argument("--expert_group_name", type=str)
    parser.add_argument("-g", "--group_name", type=str)
    parser.add_argument("-s", "--n_scenario", type=int, choices=[200, 1000],
                        default=1000, help="number of scenario")
    parser.add_argument("--sdx", type=int, choices=range(1, 4), default=1,
                        help="pre-generated scenario set index.")

    args = parser.parse_args()
    run_NER_SPSP_CVaR('dissertation', args.nr_strategy, args.nr_param,
                      args.expert_group_name, args.group_name, args.n_scenario,
                      args.sdx, '20050103', '20181228')
