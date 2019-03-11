# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

"""

import platform
import os
import datetime as dt

# data storage
node_name = platform.node()
EXP_NAME = "dissertation"

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


if EXP_NAME == "stocksp_cor15":
    DATA_DIR = os.path.join(PROJECT_DIR, "data", EXP_NAME)

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
        "TAIEX_2005_largested_market_cap_"
        "scenario-set-idx{sdx}_"
        "{scenario_start_date}_"
        "{scenario_end_date}_"
        "Mc{n_symbol}_"
        "h{rolling_window_size}_s{n_scenario}.nc"
    )

    SYMBOL_SCENARIO_NAME_FORMAT = (
        "TAIEX_2005_largested_market_cap_"
        "scenario-set-idx{sdx}_"
        "{scenario_start_date}_"
        "{scenario_end_date}_"
        "symbol{symbol}_"
        "h{rolling_window_size}_s{n_scenario}.nc"
    )

    SCENARIO_START_DATE = dt.date(2005, 1, 3)
    SCENARIO_END_DATE = dt.date(2017, 12, 29)

if EXP_NAME == "dissertation":
    DATA_DIR = os.path.join(PROJECT_DIR, "data", EXP_NAME)

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
    EXP_END_DATE = dt.date(2018, 12, 31)
    BUY_TRANS_FEE = 0.001425
    SELL_TRANS_FEE = 0.004425
    DISCOUNT_FACTOR = 0.006
    REPORT_DIR = os.path.join(DATA_DIR, "report")

    # scenario
    # SCENARIO_SET_DIR = TMP_DIR
    SCENARIO_SET_DIR = os.path.join(DATA_DIR, "scenario")
    SCENARIO_NAME_FORMAT = (
        "TAIEX_2005_market_cap_"
        "scenario-set-idx{sdx}_"
        "{scenario_start_date}_"
        "{scenario_end_date}_"
        "Mc{n_symbol}_"
        "h{rolling_window_size}_s{n_scenario}.nc"
    )

    GROUP_SCENARIO_NAME_FORMAT = (
        "gid_{group_id}_"
        "scenario-sdx{sdx}_"
        "{scenario_start_date}_"
        "{scenario_end_date}_"
        "h{rolling_window_size}_s{n_scenario}.nc"
    )

    SCENARIO_START_DATE = dt.date(2005, 1, 3)
    SCENARIO_END_DATE = dt.date(2018, 12, 31)
