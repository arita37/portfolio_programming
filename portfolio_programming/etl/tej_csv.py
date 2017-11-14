# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

transforming TEJ's stock csv to pandas panel data
"""
from datetime import date
from time import time
from glob import glob
import os
import numpy as np
import pandas as pd


def cp950_to_utf8(data):
    """ utility function in parsing csv """
    return data.strip().decode('cp950')


def data_strip(data):
    """ utility function in parsing csv """
    return data.strip()


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
    pandas.frame

    """
    t0 = time()

    for rdx, symbol in enumerate(symbols):
        csv_name = os.path.join(csv_dir, "{}.csv".format(symbol))
        print(csv_name)
        df = pd.read_csv(open(csv_name),
                         index_col=("year_month_day",),
                         parse_dates=True,
                         dtype={
                             'symbol': str,
                             'abbreviation': str,
                             'year_month_day': date,
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
                             'symbol': data_strip,
                             'abbreviation': cp950_to_utf8,
                         },
                         )

        # output data file path
        fout_path = os.path.join(df_dir, '{}_panel.pkl'.format(symbol))
        df.to_pickle(fout_path)

        print("[{}/{}]{}.csv to panel OK, {:.3f} secs".format(
            rdx + 1, len(symbols), symbol, time() - t0))

    print("csv_to_pkl OK, {:.3f} secs".format(time() - t0))


def dataframe_to_panel(symbols, df_dir):
    """
    aggregating and trimming data to a panel file
    """
    t0 = time()
    start_date = date(2004, 1, 1)
    end_date = date(2017, 6, 30)

    # load first df to read the periods
    fin_path = os.path.join(df_dir, "{}_df.pkl".format(symbols[0]))
    df = pd.read_pickle(fin_path)

    # get trans_dates and columns
    trans_dates = df[start_date:end_date].index
    trans_dates.name = 'trans_dates'
    minor_indices = ['close_price', 'simple_roi']

    # setting panel
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

        # note: pnl.loc[:, symbol, :], shape: (columns, n_exp_period)
        pnl.loc[:, symbol, :] = trimmed_df.ix[:, ('close_price',
                                                  'simple_roi')].T

        print("[{}/{}] {} load to panel OK, {:.3f} secs".format(
            sdx, len(symbols), symbol, time() - t1))

    # # fill na with 0
    # pnl = pnl.fillna(0)

    # output data file path
    fout_path = os.path.join(df_dir,
                             'TAIEX_2005_largest50cap_panel.pkl')
    pnl.to_pickle(fout_path)

    print("all exp_symbols load to panel OK, {:.3f} secs".format(time() - t0))


def run_tej_csv_to_panel():
    import portfolio_programming as pp
    import json

    symbol_file = os.path.join(pp.DATA_DIR,
                               'TAIEX_20050103_50largest_listed_market_cap.json')

    with open(symbol_file) as fin:
        symbols = json.load(fin)

    csv_dir = os.path.join(pp.DATA_DIR, 'tej_csv')
    df_dir = pp.TMP_DIR

    # run
    tej_csv_to_df(symbols, csv_dir, df_dir)


if __name__ == '__main__':
    pass
    run_tej_csv_to_panel()
