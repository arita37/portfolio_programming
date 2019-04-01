# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import datetime as dt
import logging
import os
import sys

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.simulation.wp_eg import (EGPortfolio,
                                                    EGAdaptivePortfolio,
                                                    ExpPortfolio,
                                                    ExpAdaptivePortfolio,
                                                    NIRExpPortfolio)


def run_eg(eta, exp_type, group_name, exp_start_date, exp_end_date):
    group_symbols = pp.GROUP_SYMBOLS
    if group_name not in group_symbols.keys():
        raise ValueError('Unknown group name:{}'.format(group_name))
    symbols = group_symbols[group_name]
    n_symbol = len(symbols)

    market = group_name[:2]
    if market == "TW":
        roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    elif market == "US":
        roi_xarr = xr.open_dataarray(pp.DJIA_2005_NC)

    rois = roi_xarr.loc[exp_start_date:exp_end_date, symbols, 'simple_roi']

    initial_wealth = 100
    initial_weights = xr.DataArray(
        np.ones(n_symbol) / n_symbol,
        dims=('symbol',),
        coords=(symbols,)
    )
    if exp_type == 'eg':
        exp_class = EGPortfolio
    elif exp_type == 'exp':
        exp_class = ExpPortfolio
    elif exp_type == 'nir':
        exp_class = NIRExpPortfolio
    else:
        raise ValueError('unknown exp_type:', exp_type)

    obj = exp_class(
        eta,
        group_name,
        symbols,
        rois,
        initial_weights,
        initial_wealth,
        start_date=exp_start_date,
        end_date=exp_end_date,
    )
    obj.run()


def run_eg_adaptive(group_name, exp_type, exp_start_date, exp_end_date,
                    beta=None):
    group_symbols = pp.GROUP_SYMBOLS
    if group_name not in group_symbols.keys():
        raise ValueError('Unknown group name:{}'.format(group_name))
    symbols = group_symbols[group_name]
    n_symbol = len(symbols)

    market = group_name[:2]
    if market == "TW":
        roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    elif market == "US":
        roi_xarr = xr.open_dataarray(pp.DJIA_2005_NC)

    rois = roi_xarr.loc[exp_start_date:exp_end_date, symbols, 'simple_roi']

    initial_wealth = 100
    initial_weights = xr.DataArray(
        np.ones(n_symbol) / n_symbol,
        dims=('symbol',),
        coords=(symbols,)
    )

    if exp_type == 'eg':
        exp_class = EGAdaptivePortfolio
    elif exp_type == 'exp':
        exp_class = ExpAdaptivePortfolio
    else:
        raise ValueError('unknown exp_type:', exp_type)

    obj = exp_class(
        group_name,
        symbols,
        rois,
        initial_weights,
        initial_wealth,
        start_date=exp_start_date,
        end_date=exp_end_date,
        beta=beta
    )
    obj.run()


def get_eg_report(exp_type, report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
    import pickle
    import pandas as pd
    import csv
    import arch.bootstrap.multiple_comparison as arch_comp

    if exp_type not in ('eg', 'exp', 'nir'):
        raise ValueError('unknown exp_type:', exp_type)

    group_names = pp.GROUP_SYMBOLS.keys()
    output_file = os.path.join(pp.TMP_DIR, "{}_stat.csv".format(exp_type))
    with open(output_file, "w", newline='') as csv_file:
        fields = [
            "simulation_name",
            "eta",
            "group_name",
            "start_date",
            "end_date",
            "n_data",
            "cum_roi",
            "annual_roi",
            "roi_mu",
            "std",
            "skew",
            "ex_kurt",
            "Sharpe",
            "Sortino_full",
            "Sortino_partial",
            "SPA_c"
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        if exp_type == 'eg':
            etas = ["{:.1f}".format(eta / 10) for eta in range(1, 10 + 1)]
            etas.extend(
                ["{:.2f}".format(eta) for eta in (0.01, 0.02, 0.03, 0.05)])
            etas.extend(["{:.1f}".format(eta) for eta in (2, 3, 4)])

            report_pkls = [
                (group_name,
                 "report_EG_{}_{}_20050103_20181228.pkl".format(
                     eta, group_name)
                 )
                for eta in etas
                for gdx, group_name in enumerate(group_names)
            ]
            report_pkls.extend([
                (group_name,
                 "report_EG_Adaptive_{}_20050103_20181228.pkl".format(
                     group_name)
                 )
                for group_name in group_names
            ])
        elif exp_type == 'exp':
            etas = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
            report_pkls = [
                (group_name,
                 "report_Exp_{:.2f}_{}_20050103_20181228.pkl".format(
                     eta, group_name)
                 )
                for eta in etas
                for gdx, group_name in enumerate(group_names)
            ]

        for group_name, report_name in report_pkls:
            report_file = os.path.join(pp.WEIGHT_PORTFOLIO_REPORT_DIR,
                                       report_name)
            rp = pd.read_pickle(report_file)
            # SPA value
            if "SPA_c" not in rp.keys():
                rois = rp['decision_xarr'].loc[:, :, 'wealth'].sum(
                    axis=1).to_series().pct_change()
                rois[0] = 0

                spa_value = 0
                for _ in range(3):
                    spa = arch_comp.SPA(rois.values, np.zeros(rois.size),
                                        reps=1000)
                    spa.seed(np.random.randint(0, 2 ** 31 - 1))
                    spa.compute()
                    # preserve the worse p_value
                    if spa.pvalues[1] > spa_value:
                        spa_value = spa.pvalues[1]
                rp['SPA_c'] = spa_value
                # write back to file
                with open(report_file, 'wb') as fout:
                    pickle.dump(rp, fout, pickle.HIGHEST_PROTOCOL)

            eta_value = rp.get('eta', 'adaptive')

            writer.writerow(
                {
                    "simulation_name": rp["simulation_name"],
                    "group_name": group_name,
                    "eta": eta_value,
                    "start_date": rp['exp_start_date'].strftime("%Y-%m-%d"),
                    "end_date": rp['exp_end_date'].strftime("%Y-%m-%d"),
                    "n_data": rp['n_exp_period'],
                    "cum_roi": rp['cum_roi'],
                    "annual_roi": rp['annual_roi'],
                    "roi_mu": rp['daily_mean_roi'],
                    "std": rp['daily_std_roi'],
                    "skew": rp['daily_skew_roi'],
                    "ex_kurt": rp['daily_ex-kurt_roi'],
                    "Sharpe": rp['Sharpe'],
                    "Sortino_full": rp['Sortino_full'],
                    "Sortino_partial": rp['Sortino_partial'],
                    "SPA_c": rp['SPA_c']
                }
            )
            print(
                "{}_{} {}, cum_roi:{:.2%}".format(
                    exp_type, eta_value, group_name, rp['cum_roi']
                )
            )


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(filename)15s %(levelname)10s %(asctime)s\n'
               '%(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--simulation", default=False,
                        action='store_true',
                        help="EG experiment")
    parser.add_argument("--exp_type", type=str,
                        help="experiment type: eg or exp or nir")
    parser.add_argument("--eta", type=float,
                        help="learning rate")
    parser.add_argument("--adaptive", default=False,
                        action='store_true',
                        help="EG adaptive experiment")
    parser.add_argument("--beta", type=float,
                        help="price relative ratio")
    parser.add_argument("--stat", default=False,
                        action='store_true',
                        help="EG experiment statistics")
    parser.add_argument("-g", "--group_name", type=str,
                        help="target group")

    args = parser.parse_args()
    if args.simulation:
        if args.group_name:
            print(args.eta, args.exp_type, args.group_name)
            run_eg(args.eta, args.exp_type, args.group_name,
                   dt.date(2005, 1, 1), dt.date(2018, 12, 28))
        else:
            import multiprocess as mp

            n_cpu = mp.cpu_count() // 2 if mp.cpu_count() >= 2 else 1
            pool = mp.Pool(processes=n_cpu)
            results = [pool.apply_async(run_eg,
                                        (args.eta, args.exp_type, group_name,
                                         dt.date(2005, 1, 1),
                                         dt.date(2018, 12, 28))
                                        )
                       for group_name in pp.GROUP_SYMBOLS.keys()
                       ]
            [result.wait() for result in results]
            pool.close()
            pool.join()

    if args.adaptive:
        import multiprocess as mp
        n_cpu = mp.cpu_count() // 2 if mp.cpu_count() >= 2 else 1
        pool = mp.Pool(processes=n_cpu)

        results = [pool.apply_async( run_eg_adaptive,
                                    (group_name, args.exp_type,
                                     dt.date(2005, 1, 1),
                                     dt.date(2018, 12, 28))
                                    )
                   for group_name in pp.GROUP_SYMBOLS.keys()
                   ]
        [result.wait() for result in results]
        pool.close()
        pool.join()

    if args.stat:
        get_eg_report(args.exp_type)
