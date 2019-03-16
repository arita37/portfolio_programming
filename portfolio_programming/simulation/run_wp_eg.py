# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import datetime as dt
import logging
import sys

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.simulation.wp_eg import EGPortfolio


def run_eg(eta, group_name, exp_start_date, exp_end_date):
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

    obj = EGPortfolio(
        eta,
        group_name,
        symbols,
        rois,
        initial_weights,
        initial_wealth,
        start_date=exp_start_date,
        end_date=exp_end_date
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
                        help="EG experiment")
    parser.add_argument("--eta", type=float,
                        help="learning rate")

    args = parser.parse_args()
    if args.simulation:
        for group_name in pp.GROUP_SYMBOLS.keys():
            run_eg(args.eta, 'TWG1',
                   dt.date(2005, 1, 1), dt.date(2018, 12, 28))
