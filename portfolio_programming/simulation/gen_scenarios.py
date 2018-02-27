# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

https://stackoverflow.com/questions/37480402/how-to-correctly-import-modules-on-engines-in-jupyter-notebook-for-parallel-proc
https://stackoverflow.com/questions/18570071/import-custom-modules-on-ipython-parallel-engines-with-sync-imports

https://stackoverflow.com/questions/18086299/real-time-output-from-engines-in-ipython-parallel

https://ask.helplib.com/parallel-processing/post_5405801

https://stackoverflow.com/questions/23145650/how-to-setup-ssh-tunnel-for-ipython-cluster-ipcluster/31479269
"""

import glob
import json
import os
from time import time, sleep

import ipyparallel as ipp
import numpy as np
import pandas as pd

import portfolio_programming as pp
from portfolio_programming.sampling.moment_matching import (
    heuristic_moment_matching as HeMM)


def generating_scenarios_pnl(scenario_set_idx,
                             scenario_start_date,
                             scenario_end_date,
                             n_stock,
                             rolling_window_size,
                             n_scenario,
                             retry_cnt=5,
                             print_interval=5):
    """
    generating scenarios panel

    Parameters:
    ------------------
    n_stock: positive integer, number of stocks in the candidate symbols
    rolling_window_size: positive integer, number of historical periods
    n_scenario: integer, number of scenarios to generating
    etry_cnt: positive integer, maximum retry of scenarios
    """

    # import for parallel processing
    # import os
    # from time import time
    # import pandas as pd
    # import portfolio_programming as pp
    # from portfolio_programming.sampling.moment_matching import (
    #     heuristic_moment_matching as HeMM)

    t0 = time()

    # scenario dir
    if not os.path.exists(pp.SCENARIO_SET_DIR):
        os.makedirs(pp.SCENARIO_SET_DIR)

    scenario_file = pp.SCENARIO_NAME_FORMAT.format(
        sdx=scenario_set_idx,
        scenario_start_date=scenario_start_date.strftime("%y%m%d"),
        scenario_end_date=scenario_end_date.strftime("%y%m%d"),
        n_stock=n_stock,
        rolling_window_size=rolling_window_size,
        n_scenario=n_scenario
    )

    scenario_path = os.path.join(pp.SCENARIO_SET_DIR, scenario_file)
    if os.path.exists(scenario_path):
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
    sc_trans_dates = trans_dates[sc_start_idx: sc_end_idx + 1]
    n_sc_period = len(sc_trans_dates)

    # estimating moments and correlation matrix
    est_moments = pd.DataFrame(np.zeros((n_stock, 4)), index=candidate_symbols)

    # output scenario panel, shape: (n_sc_period, n_stock, n_scenario)
    scenario_pnl = pd.Panel(
        np.zeros((n_sc_period, n_stock, n_scenario)),
        items=sc_trans_dates,
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
                        print("relaxing max err: {}_{}_max_mom_err:{}, "
                              "max_corr_err{}".format(sc_date, parameters,
                                                      max_moment_err,
                                                      max_corr_err))
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
        if tdx % print_interval == 0:
            print("{}_{}, {} [{}/{}] {} OK, {:.4f} secs".format(
                platform.node(),
                os.getpid(),
                sc_date.strftime("%Y%m%d"),
                tdx + 1,
                n_sc_period,
                parameters,
                time() - t1))

    # write scenario
    scenario_pnl.to_pickle(scenario_path)

    msg = ("generating scenarios {} OK, {:.3f} secs".format(
        parameters, time() - t0))
    print(msg)
    return msg


def _all_scenario_names():
    """
    SCENARIO_NAME_FORMAT = "TAIEX_2005_largested_market_cap_" \
                       "scenario-set-idx{sdx}_" \
                       "{scenario_start_date}_" \
                       "{scenario_end_date}_" \
                       "Mc{n_stock}_" \
                       "h{rolling_window_size}_s{n_scenario}.pkl"
    """
    # set_indices = (1, 2, 3)
    set_indices = (1,)
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


def wait_watching_stdout(ar, dt=1, truncate=1000):
    from IPython.display import clear_output
    import sys
    import platform
    import os
    from time import sleep

    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        print('-' * 50)
        print("node:{} pid:{} {:.3f}s elapsed".format(
            platform.node(),
            os.getpid(),
            ar.elapsed))

        for stdout in ar.stdout:
            if stdout:
                print(stdout[-truncate:])
        sys.stdout.flush()
        sleep(dt)


def dispatch_scenario_names(scenario_set_dir=pp.SCENARIO_SET_DIR):
    from IPython.display import clear_output


    unfinished_names = checking_existed_scenario_names(scenario_set_dir)
    print("number of unfinished scenario: {}".format(len(unfinished_names)))
    params = unfinished_names.values()

    # task interface
    rc = ipp.Client(profile='ssh')
    dv = rc[:]
    dv.use_dill()

    with dv.sync_imports():
        import sys
        import platform
        import os
        import portfolio_programming.simulation.gen_scenarios

    def name_pid():
        return "node:{}, pid:{}".format(platform.node(), os.getpid())

    infos = dv.apply_sync(name_pid)
    for info in infos:
        print(info)

    lbv = rc.load_balanced_view()
    ar = lbv.map_async(
        lambda x:portfolio_programming.simulation.gen_scenarios.generating_scenarios_pnl(*x),
            params)

    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        for stdout in ar.stdout:
            if stdout:
                print(stdout[-50:])
        sys.stdout.flush()
        sleep(1)


if __name__ == '__main__':
    # generating_scenarios_pnl(1, pp.SCENARIO_START_DATE, pp.SCENARIO_END_DATE,
    #                          5, 50)
    dispatch_scenario_names()
