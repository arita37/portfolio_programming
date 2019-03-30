# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""
import datetime as dt
import logging
import os
import sys

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.simulation.wp_poly import PolynomialPortfolio


def run_poly(poly_power, group_name, exp_start_date, exp_end_date):
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

    obj = PolynomialPortfolio(
        poly_power,
        group_name,
        symbols,
        rois,
        initial_weights,
        initial_wealth,
        start_date=exp_start_date,
        end_date=exp_end_date,
        print_interval=10
    )
    obj.run()


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
                        help="polynomial forecaster experiment")
    parser.add_argument("--poly", type=float,
                        help="polynomial power")
    parser.add_argument("-g", "--group_name", type=str,
                        help="target group")

    args = parser.parse_args()
    if args.simulation:
        if args.group_name:
            run_poly(args.poly, args.group_name,
                    dt.date(2005, 1, 1), dt.date(2018, 12, 28))

        else:
            import multiprocess as mp
            n_cpu = mp.cpu_count() // 2 if mp.cpu_count() >= 2 else 1
            pool = mp.Pool(processes=n_cpu)
            results = [pool.apply_async(run_poly,
                                        (args.poly, group_name,
                                         dt.date(2005, 1, 1), dt.date(2018, 12, 28))
                                        )
                       for group_name in pp.GROUP_SYMBOLS.keys()
                       ]
            [result.wait() for result in results]
            pool.close()
            pool.join()