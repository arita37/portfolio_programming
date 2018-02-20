# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

transforming TEJ's stock csv to pandas panel data
"""

import datetime as dt
from time import time
import os
import numpy as np
import pandas as pd


def tej_csv_to_df(symbols, csv_dir, df_dir):
    """
    extract data from csv

    Parameters:
    --------------
    symbols : list
        list of stock symbols given in data directory
    csv_dir : string
        csv file directory of symbols
    df_dir : string
        pandas dataframe output directory

    Returns:
    --------------
    list of pandas.dataframe files

    """
    t0 = time()

    for rdx, symbol in enumerate(symbols):
        csv_name = os.path.join(csv_dir, "{}.csv".format(symbol))
        df = pd.read_csv(open(csv_name),
                         index_col='year_month_day',
                         parse_dates=True,  # parse the index column

                         dtype={
                             'symbol': str,
                             'abbreviation': str,
                             'year_month_day': str,
                             'open_price': np.float,
                             'high_price': np.float,
                             'low_price': np.float,
                             'close_price': np.float,
                             'volume_1000_shares': np.float,
                             'value_1000_dollars': np.float,
                             'simple_roi_%': np.float,
                             'turnover_ratio_%': np.float,
                             'market_value_million_dollars': np.float,
                             'continuous_roi_%': np.float,
                         },
                         converters={
                             'symbol': lambda x: x.strip(),
                             'abbreviation': lambda x: x.strip(),

                         },
                         )

        # output data file path
        fout_path = os.path.join(df_dir, '{}_df.pkl'.format(symbol))
        df.to_pickle(fout_path)

        print("[{}/{}]{}.csv to dataframe OK, {:.4f} secs".format(
            rdx + 1, len(symbols), symbol, time() - t0))

    print("csv_to_to OK, elapsed {:.4f} secs".format(time() - t0))


def dataframe_to_panel(symbols, df_dir, start_date, end_date, fout_path):
    """
    aggregating and trimming dataframe of stock to a panel file

    Parameters:
    --------------
    symbols : list
        list of stock symbols given in data directory
    df_dir : string
        pandas dataframe  directory
    start_date, end_date : datatime.date
        the start and end dates in the panel

    Returns:
    ---------
    pandas.panel

    """
    t0 = time()

    # load first df to read the periods
    fin_path = os.path.join(df_dir, "{}_df.pkl".format(symbols[0]))
    df = pd.read_pickle(fin_path)

    # get trans_dates and columns
    trans_dates = df[start_date:end_date].index
    trans_dates.name = 'trans_dates'
    minor_indices = ['open_price', 'high_price', 'low_price', 'close_price',
                     'volume', 'simple_roi']

    # setting panel (date, symbol, indices)
    pnl = pd.Panel(
        np.zeros((len(trans_dates), len(symbols), len(minor_indices))),
        items=trans_dates,
        major_axis=symbols,
        minor_axis=minor_indices)

    for sdx, symbol in enumerate(symbols):
        t1 = time()
        # read df
        fin_path = os.path.join(df_dir, "{}_df.pkl".format(symbol))
        trimmed_df = pd.read_pickle(fin_path).loc[start_date:end_date]

        # rename columns
        trimmed_df['simple_roi_%'] /= 100.
        trimmed_df.rename(columns={r'simple_roi_%': 'simple_roi'},
                          inplace=True)

        trimmed_df.rename(columns={r'volume_1000_shares': 'volume'},
                          inplace=True)

        # note: pnl.loc[:, symbol, :], shape: (columns, n_exp_period)
        pnl.loc[:, symbol, :] = trimmed_df.ix[:, ('open_price', 'high_price',
                                                  'low_price', 'close_price',
                                                  'volume',
                                                  'simple_roi')].T

        print("[{}/{}] {} load to panel OK, {:.3f} secs".format(
            sdx + 1, len(symbols), symbol, time() - t1))

    # # fill na with 0
    # pnl = pnl.fillna(0)

    # output data file path
    pnl.to_pickle(fout_path)

    print("all exp_symbols load to panel OK, {:.3f} secs".format(time() - t0))


def run_tej_csv_to_panel():
    import portfolio_programming as pp
    import json

    with open(pp.TAIEX_SYMBOL_JSON) as fin:
        symbols = json.load(fin)

    csv_dir = os.path.join(pp.DATA_DIR, 'tej_csv')
    df_dir = pp.TMP_DIR

    # manual setting
    trim_start_date = dt.date(2005, 1, 3)
    trim_end_date = dt.date(2017, 6, 30)

    # run
    tej_csv_to_df(symbols, csv_dir, df_dir)
    dataframe_to_panel(symbols, df_dir, trim_start_date, trim_end_date,
                       pp.TAIEX_PANEL_PKL)


if __name__ == '__main__':
    run_tej_csv_to_panel()
