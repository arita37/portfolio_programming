# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import glob
import os
import platform
import json
from time import time

import numpy as np
import pandas as pd
import ipyparallel as ipp

import portfolio_programming as pp
from portfolio_programming.sampling.moment_matching import (
    heuristic_moment_matching as HeMM)


def generating_scenarios_pnl(scenario_set_idx,
                             scenario_start_date,
                             scenario_end_date,
                             n_stock,
                             rolling_window_size,
                             n_scenario=200,
                             retry_cnt=5):
    """
    generating scenarios panel
 
    Parameters:
    ------------------
    n_stock: positive integer, number of stocks in the candidate symbols
    rolling_window_size: positive integer, number of historical periods
    n_scenario: integer, number of scenarios to generating
    etry_cnt: positive integer, maximum retry of scenarios
    """
    t0 = time()

    # scenario dir
    if not os.path.exists(pp.SCENARIO_SET_DIR):
        os.makedirs(pp.SCENARIO_SET_DIR)

    scenario_file = pp.SCENARIO_NAME_FORMAT.format(
            sdx = scenario_set_idx,
            scenario_start_date=scenario_start_date.strftime("%y%m%d"),
            scenario_end_date= scenario_end_date.strftime("%y%m%d"),
            n_stock=  n_stock,
            rolling_window_size= rolling_window_size,
            n_scenario=n_scenario
    )

    scenario_path = os.path.join(pp.SCENARIO_SET_DIR, scenario_file)
    if os.path.exists( scenario_path):
        print("{} exists.".format(scenario_file))
        return

    parameters = "scenarios-set-idx{}_{}_{}_Mc{}_h{}_s{}".format(
        scenario_set_idx,
        scenario_start_date.strftime("%y%m%d"),
        scenario_end_date.strftime("%y%m%d"),
        n_stock,
        rolling_window_size,
        n_scenario,
    )

    # read symbol roi data
    # shape: (n_period, n_stock, attributes)
    risky_asset_pnl = pd.read_pickle(pp.TAIEX_PANEL_PKL)

    # symbols
    with open(pp.TAIEX_SYMBOL_JSON) as fin:
        candidate_symbols = json.load(fin)[:n_stock]

    # all trans_date
    trans_dates = risky_asset_pnl.items

    # experiment trans_dates
    sc_start_idx = trans_dates.get_loc(scenario_start_date)
    sc_end_idx = trans_dates.get_loc(scenario_end_date)
    sc_trans_dates = trans_dates[sc_start_idx: sc_end_idx+1]
    n_sc_period = len(sc_trans_dates)

    # estimating moments and correlation matrix
    est_moments = pd.DataFrame(np.zeros((n_stock, 4)), index=candidate_symbols)

    # output scenario panel, shape: (n_sc_period, n_stock, n_scenario)
    scenario_pnl = pd.Panel(
        np.zeros((n_sc_period, n_stock, n_scenario)),
        items= sc_trans_dates,
        major_axis=candidate_symbols
    )

    for tdx, sc_date in enumerate(sc_trans_dates):
        t1 = time()

        # rolling historical window indices, containing today
        est_start_idx = sc_start_idx + tdx - rolling_window_size + 1
        est_end_idx = sc_start_idx + tdx + 1
        hist_interval = trans_dates[est_start_idx:est_end_idx]

        assert len(hist_interval) == rolling_window_size
        assert hist_interval[-1] == sc_date

        # hist_data, shape: (n_stock, win_length)
        hist_data = risky_asset_pnl.loc[hist_interval,
                                        candidate_symbols,
                                        'simple_roi']

        # unbiased moments and corrs estimators
        est_moments.iloc[:, 0] = hist_data.mean(axis=1)
        est_moments.iloc[:, 1] = hist_data.std(axis=1, ddof=1)
        est_moments.iloc[:, 2] = hist_data.skew(axis=1)
        est_moments.iloc[:, 3] = hist_data.kurt(axis=1)
        est_corrs = (hist_data.T).corr("pearson")

        # generating unbiased scenario
        for error_count in range(retry_cnt):
            try:
                for error_exponent in range(-3, 0):
                    try:
                        # default moment and corr errors (1e-3, 1e-3)
                        # df shape: (n_stock, n_scenario)
                        max_moment_err = 10 ** error_exponent
                        max_corr_err = 10 ** error_exponent
                        scenario_df = HeMM(est_moments.as_matrix(),
                                           est_corrs.as_matrix(),
                                           n_scenario,
                                           False,
                                           max_moment_err,
                                           max_corr_err)
                    except ValueError as e:
                        print ("relaxing max err: {}_{}_max_mom_err:{}, "
                               "max_corr_err{}".format( sc_date, parameters,
                                max_moment_err, max_corr_err))
                    else:
                        # generating scenarios success
                        break

            except Exception as e:
                # catch any other exception
                if error_count == retry_cnt - 1:
                    raise Exception(e)
            else:
                # generating scenarios success
                break

        # store scenarios
        scenario_pnl.loc[sc_date, :, :] = scenario_df

        # clear est data
        print ("{} [{}/{}] {} OK, {:.4f} secs".format(
            sc_date.strftime("%Y%m%d"),
            tdx+1,
            n_sc_period,
            parameters,
            time() - t1))

    # write scenario
    scenario_pnl.to_pickle(scenario_path)

    print("generating scenarios {} OK, {:.3f} secs".format(
        parameters, time() - t0))




def _all_scenario_names():
    """
    SCENARIO_NAME_FORMAT = "TAIEX_2005_largested_market_cap_" \
                       "scenario-set-idx{sdx}_" \
                       "{scenario_start_date}_" \
                       "{scenario_end_date}_" \
                       "Mc{n_stock}_" \
                       "h{rolling_window_size}_s{n_scenario}.pkl"
    """
    set_indices = (1, 2, 3)
    s_date = pp.SCENARIO_START_DATE
    e_date = pp.SCENARIO_END_DATE
    n_stocks = range(5, 50 + 5, 5)
    window_sizes = range(50, 240 + 10, 10)
    n_scenarios = [200, ]

    # dict comprehension
    return {
        pp.SCENARIO_NAME_FORMAT.format(
            sdx=sdx,
            scenario_start_date=s_date,
            scenario_end_date=e_date,
            n_stock=m,
            rolling_window_size=h,
            n_scenario=s
        ): (sdx, s_date, e_date, m, h, s)
        for sdx in set_indices
        for m in n_stocks
        for h in window_sizes
        for s in n_scenarios
    }


def checking_existed_scenario_names(scenario_set_dir=None):
    """
    return unfinished experiment parameters.
    """
    if scenario_set_dir is None:
        scenario_set_dir = pp.SCENARIO_SET_DIR
    all_names = _all_scenario_names()

    existed_names = glob.glob(os.path.join(scenario_set_dir, "*.pkl"))
    for name in existed_names:
        all_names.pop(name, None)

    # unfinished params
    return all_names

#
# def checking_working_scenario_names(scenario_set_dir=pp.SCENARIO_SET_DIR,
#                                     log_file='working.pkl',
#                                     retry_cnt=5):
#     """
#     if a parameter is under working and not write to pkl,
#     it is recorded to a file
#     """
#     # get all params
#     all_names = _all_scenario_names()
#
#     # storing a dict, key: param, value: platform_name
#     log_path = os.path.join(scenario_set_dir, log_file)
#
#     # no working
#     if not os.path.exists(log_path):
#         return all_names
#
#     for retry in range(retry_cnt):
#         try:
#             # preventing multi-process write file at the same time
#             wokring_names = pd.read_pickle(log_path)
#         except IOError as e:
#             if retry == retry_cnt:
#                 raise Exception(e)
#             else:
#                 print("check working retry: {}, {}".format(retry + 1, e))
#                 time.sleep(np.random.rand() * 5)
#
#     for node, name in wokring_names.items():
#         val = all_names.pop(name, None)
#         if val:
#             print("{} under processing on {}.".format(val, node))
#
#     # unfinished params
#     return all_names

def dispatch_scenario_names(scenario_set_dir=pp.SCENARIO_SET_DIR):
    # reading working pkl
    log_path = os.path.join(scenario_set_dir)
    existed_names = checking_existed_scenario_names(scenario_set_dir)
    unfinished_names = existed_names.intersection(existed_names)
    print("number of unfinished scenario: {}".format(len(unfinished_names)))

    # task interface
    rc = ipp.Client()
    #lview = rc.load_balanced_view()
    dview = rc[:]



    while len(unfinished_names):
        # each loop we have to
        existed_names = checking_existed_scenario_names(scenario_set_dir)
        working_names = checking_working_scenario_names(scenario_set_dir,
                                                        log_file, retry_cnt)
        unfinished_names = existed_names.intersection(working_names)

        print("current unfinished scenarios: {}".format(len(unfinished_names)))

        param = unfinished_names.pop()
        _, _, stock, win, scenario, biased, _ = param.split('_')
        n_stock = int(stock[stock.rfind('m') + 1:])
        win_length = int(win[win.rfind('w') + 1:])
        n_scenario = int(scenario[scenario.rfind('s') + 1:])

        # log  parameter to file
        if not os.path.exists(log_path):
            working_dict = {}
        else:
            for retry in xrange(retry_cnt):
                try:
                    # preventing multi-process write file at the same time
                    working_dict = pd.read_pickle(log_path)
                except IOError as e:
                    if retry == retry_cnt - 1:
                        raise Exception(e)
                    else:
                        print("working retry: {}, {}".format(retry + 1, e))
                        time.sleep(np.random.rand() * 5)

        working_dict[param] = platform.node()
        for retry in xrange(retry_cnt):
            try:
                # preventing multi-process write file at the same time
                pd.to_pickle(working_dict, log_path)
            except IOError as e:
                if retry == retry_cnt - 1:
                    raise Exception(e)
                else:
                    print("working retry: {}, {}".format(retry + 1, e))
                    time.sleep(np.random.rand() * 5)

        # generating scenarios
        try:
            print("gen scenario: {}".format(param))
            generating_scenarios_pnl(n_stock, win_length, n_scenario, bias)
        except Exception as e:
            print
            param, e
        finally:
            for retry in xrange(retry_cnt):
                try:
                    # preventing multi-process write file at the same time
                    working_dict = pd.read_pickle(log_path)
                except IOError as e:
                    if retry == retry_cnt - 1:
                        raise Exception(e)
                    else:
                        print("working retry: {}, {}".format(retry + 1, e))
                        time.sleep(np.random.rand() * 5)

            if param in working_dict.keys():
                del working_dict[param]
            else:
                print("can't find {} in working dict.".format(param))
            for retry in xrange(retry_cnt):
                try:
                    # preventing multi-process write file at the same time
                    pd.to_pickle(working_dict, log_path)
                except IOError as e:
                    if retry == retry_cnt - 1:
                        raise Exception(e)
                    else:
                        print("finally retry: {}, {}".format(retry + 1, e))
                        time.sleep(2)


def read_working_parameters():
    scenario_path = os.path.join(EXP_SP_PORTFOLIO_DIR, 'scenarios')
    log_file = 'working.pkl'
    file_path = os.path.join(scenario_path, log_file)

    if not os.path.exists(file_path):
        print("{} not exists.".format(file_path))
    else:
        working_dict = pd.read_pickle(file_path)
        for param, node in working_dict.items():
            print
            param, node


if __name__ == '__main__':
    generating_scenarios_pnl(1, pp.SCENARIO_START_DATE, pp.SCENARIO_END_DATE,
                             5, 50)
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("-b", "--bias", action='store_true')
    # group.add_argument("-u", "--unbias", action='store_true')
    # args = parser.parse_args()
    # if args.bias:
    #     dispatch_scenario_names(bias_estimator=True)
    # elif args.unbias:
    #     dispatch_scenario_names(bias_estimator=False)
