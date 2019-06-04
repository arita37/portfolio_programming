# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import platform
import os
import datetime as dt
import json

# data storage
node_name = platform.node()
exp_name = "rm_spsp"

def valid_exp_name(exp_name):
    if exp_name not in ('rm_spsp', 'spsp'):
        raise ValueError('unknown exp_name:', exp_name)


if node_name == "X220":
    # windows 10
    PROJECT_DIR = r"C:\Users\chen1\Documents\workspace_pycharm\portfolio_programming"
    TMP_DIR = r"e:"

elif node_name == "tanh2-480s":
    # windows 10
    PROJECT_DIR = r"C:\Users\chen1\PycharmProjects\portfolio_programming"
    TMP_DIR = r"e:"
else:
    # ubuntu linux 16.04
    PROJECT_DIR = r"/home/chenhh/workspace_pycharm/portfolio_programming"
    TMP_DIR = r"/tmp"

if exp_name == "spsp":
    DATA_DIR = os.path.join(PROJECT_DIR, "data", exp_name)

    # candidate_symbol sets
    TAIEX_2005_MKT_CAP_50_SYMBOL_JSON = os.path.join(
        DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap.json"
    )

    # netCDF file
    TAIEX_2005_MKT_CAP_NC = os.path.join(
        DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap_xarray.nc"
    )

    # solver
    PROG_SOLVER = "cplex"

    # simulation
    EXP_START_DATE = dt.date(2005, 1, 3)
    EXP_END_DATE = dt.date(2014, 12, 31)
    BUY_TRANS_FEE = 0.001425
    SELL_TRANS_FEE = 0.004425
    REPORT_DIR = os.path.join(DATA_DIR, "report")

    # scenario
    # SCENARIO_SET_DIR = TMP_DIR
    SCENARIO_SET_DIR = os.path.join(DATA_DIR, "scenario")
    SCENARIO_NAME_FORMAT = (
        "{group_name}_"
        "Mc{n_symbol}_"
        "h{rolling_window_size}_"
        "s{n_scenario}_"
        "sdx{sdx}_"
        "{scenario_start_date}_"
        "{scenario_end_date}.nc"
    )


    SCENARIO_START_DATE = dt.date(2005, 1, 3)
    SCENARIO_END_DATE = dt.date(2017, 12, 29)

if exp_name == "rm_spsp":
    DATA_DIR = os.path.join(PROJECT_DIR, "data", exp_name)

    # candidate_symbol sets
    TAIEX_2005_MKT_CAP_50_SYMBOL_JSON = os.path.join(
        DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap.json"
    )

    DJIA_2005_SYMBOL_JSON = os.path.join(DATA_DIR, "DJIA_exp_symbols.json")

    # netCDF file
    TAIEX_2005_MKT_CAP_NC = os.path.join(
        DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap_xarray.nc"
    )

    DJIA_2005_NC = os.path.join(DATA_DIR, "DJIA_exp_xarray.nc")

    # solver
    PROG_SOLVER = "cplex"

    # simulation
    EXP_START_DATE = dt.date(2005, 1, 3)
    EXP_END_DATE = dt.date(2018, 12, 28)
    BUY_TRANS_FEE = 0.001425
    SELL_TRANS_FEE = 0.004425
    DISCOUNT_FACTOR = 0.006
    REPORT_DIR = os.path.join(DATA_DIR, "report")
    WEIGHT_PORTFOLIO_REPORT_DIR = os.path.join(DATA_DIR,
                                               'report_weight_portfolio')
    NRSPSPCVaR_DIR = os.path.join(DATA_DIR,
                                  'report_nrspsp_cvar')

    # scenario
    # SCENARIO_SET_DIR = TMP_DIR
    if node_name in ('X220', "tanh2-480s", 'eva00'):
        SCENARIO_SET_DIR = os.path.join(DATA_DIR, "scenario")
    else:
        SCENARIO_SET_DIR = r'/home/chenhh/workspace_pycharm_local/scenario'

    SCENARIO_NAME_FORMAT = (
        "{group_name}_"
        "Mc{n_symbol}_"
        "h{rolling_window_size}_"
        "s{n_scenario}_"
        "sdx{sdx}_"
        "{scenario_start_date}_"
        "{scenario_end_date}.nc"
    )

    tw_symbols = json.load(open(TAIEX_2005_MKT_CAP_50_SYMBOL_JSON))
    us_symbols = json.load(open(DJIA_2005_SYMBOL_JSON))
    GROUP_SYMBOLS = {
        '{}G{}'.format(mkt, idx+1) : symbols[sdx:sdx+5]
        for mkt, symbols in zip(['TW', 'US'], [tw_symbols, us_symbols])
        for idx, sdx in enumerate(range(0, 30, 5))
    }
    SCENARIO_START_DATE = dt.date(2005, 1, 3)
    SCENARIO_END_DATE = dt.date(2018, 12, 28)

