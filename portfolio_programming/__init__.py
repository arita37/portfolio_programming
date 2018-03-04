# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import platform
import os
import datetime as dt

# data storage
node_name = platform.node()

if node_name == 'X220':
    # windows 10
    PROJECT_DIR = r'C:\Users\chen1\Documents\workspace_pycharm\portfolio_programming'
    TMP_DIR = r'e:'
else:
    # ubuntu linux 16.04
    PROJECT_DIR = r'/home/chenhh/workspace_pycharm/portfolio_programming'
    TMP_DIR = r'/tmp'

DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# candidate_symbol sets
TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON = os.path.join(DATA_DIR,
                                 'TAIEX_20050103_50largest_listed_market_cap.json')

# netCDF file
TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC = os.path.join(DATA_DIR,
                             'TAIEX_20050103_50largest_listed_market_cap_xarray.nc')

# solver
PROG_SOLVER = 'cplex'

# simulation
EXP_START_DATE = dt.date(2005, 1, 3)
EXP_END_DATE = dt.date(2014, 12, 31)
BUY_TRANS_FEE = 0.001425
SELL_TRANS_FEE = 0.004425
REPORT_DIR = os.path.join(DATA_DIR, 'report')

# scenario
# SCENARIO_SET_DIR = TMP_DIR
SCENARIO_SET_DIR = os.path.join(DATA_DIR, 'scenario')
SCENARIO_NAME_FORMAT = "TAIEX_2005_largested_market_cap_" \
                       "scenario-set-idx{sdx}_" \
                       "{scenario_start_date}_" \
                       "{scenario_end_date}_" \
                       "Mc{n_stock}_" \
                       "h{rolling_window_size}_s{n_scenario}.nc"
SCENARIO_START_DATE = dt.date(2005, 1, 3)
SCENARIO_END_DATE = dt.date(2014, 12, 31)

