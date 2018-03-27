# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import datetime as dt
import glob
import json
import os
from time import time

import numpy as np
import pandas as pd
import xarray as xr

import portfolio_programming as pp


def yahoo_us_csv_to_df(symbols, csv_dir, df_dir):
    """
    extract data from csv, and transforms the csv to dataframe
    csv head: Date,Open,High,Low,Close,Adj Close,Volume

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
        csvs = glob.glob(os.path.join(csv_dir, "{}_yahoo_*.csv".format(symbol)))
        if len(csvs) > 1:
            raise ValueError("{} has more than one csv files".format(symbol))
        csv_name = csvs[0]
        print(csv_name)
        df = pd.read_csv(open(csv_name),
                         index_col='Date',
                         parse_dates=True,  # parse the index column
                         dtype={
                             # 'symbol': str,
                             # 'abbreviation': str,
                             'Date': str,
                             'Open': np.float,
                             'High': np.float,
                             'Low': np.float,
                             'Close': np.float,
                             'Adj Close': np.float,
                             'Volume': np.float,
                         },
                         )
        df.rename(columns={
            r'Open': 'open_price',
            r'High': 'high_price',
            r'Low': 'low_price',
            r'Close': 'close_price',
            r'Adj Close': 'adj_close_price',
            r'Volume': 'volume'},
            inplace=True)

        # output data file path
        fout_path = os.path.join(df_dir, '{}_df.pkl'.format(symbol))
        df.to_pickle(fout_path)

        print("[{}/{}]{} csv to dataframe OK dates:{}_{}, {:.4f} secs".format(
            rdx + 1, len(symbols), symbol,
            df.index[0], df.index[-1], time() - t0))

    print("csv_to_to OK, elapsed {:.4f} secs".format(time() - t0))


def dataframe_to_xarray(symbols, df_dir, start_date, end_date, fout_path):
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
    xarray.DataArray

    """
    t0 = time()

    # load first df to read the periods
    fin_path = os.path.join(df_dir, "{}_df.pkl".format(symbols[0]))
    df = pd.read_pickle(fin_path)

    # get trans_dates and columns
    trans_dates = df[start_date:end_date].index
    trans_dates.name = 'trans_date'
    minor_indices = ['open_price', 'high_price', 'low_price', 'close_price',
                     'adj_close_price', 'volume']

    # setting xarray (date, symbol, indices)
    xarr = xr.DataArray(
        np.zeros((len(trans_dates), len(symbols), len(minor_indices))),
        dims=('trans_date', 'symbol', 'data'),
        coords=[trans_dates, symbols, minor_indices]
    )

    for sdx, symbol in enumerate(symbols):
        t1 = time()
        # read df
        fin_path = os.path.join(df_dir, "{}_df.pkl".format(symbol))
        trimmed_df = pd.read_pickle(fin_path).loc[start_date:end_date]

        # rename columns
        dates = trimmed_df.index
        xarr.loc[dates, symbol, :] = trimmed_df.loc[
            dates, ('open_price', 'high_price',
                    'low_price', 'close_price', 'adj_close_price',
                    'volume')]

        print("[{}/{}] {} load to xarray OK, {:.3f} secs".format(
            sdx + 1, len(symbols), symbol, time() - t1))

    # # fill na with 0
    # pnl = pnl.fillna(0)

    # output data file path
    xarr.to_netcdf(fout_path)

    print("all exp_symbols load to xarray OK, {:.3f} secs".format(time() - t0))


def run_yahoo_us_csv_to_xarray(trim_start_date=dt.date(1990, 1, 2),
                               trim_end_date=dt.date(2018, 2, 28)
                               ):
    name = 'DJIA_symbols_20170901.json'
    with open(os.path.join(pp.DATA_DIR, name)) as fin:
        symbols = json.load(fin)

    csv_dir = os.path.join(pp.DATA_DIR, 'yahoo_us_csv')
    df_dir = pp.TMP_DIR

    yahoo_us_csv_to_df(symbols, csv_dir, df_dir)
    dataframe_to_xarray(symbols, df_dir, trim_start_date, trim_end_date,
                        os.path.join(pp.TMP_DIR, "DJIA_symbols_20170901.nc"))


if __name__ == '__main__':
    run_yahoo_us_csv_to_xarray()
