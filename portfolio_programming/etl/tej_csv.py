# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3

transforming TEJ's stock csv to pandas panel data
"""

import datetime as dt
import json
import os
from time import time

import numpy as np
import pandas as pd
import xarray as xr

import portfolio_programming as pp


def tej_csv_to_df(symbols, csv_dir, df_dir):
    """
    extract data from csv, and transforms the csv to dataframe

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
                             # 'symbol': str,
                             # 'abbreviation': str,
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

        print("[{}/{}]{} csv to dataframe OK dates:{}_{}, {:.4f} secs".format(
            rdx + 1, len(symbols), symbol,
            df.index[0],
            df.index[-1],
            time() - t0))

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
                     'volume', 'simple_roi']

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
        trimmed_df['simple_roi_%'] /= 100.
        trimmed_df.rename(columns={r'simple_roi_%': 'simple_roi'},
                          inplace=True)

        trimmed_df.rename(columns={r'volume_1000_shares': 'volume'},
                          inplace=True)

        xarr.loc[dates, symbol, :] = trimmed_df.loc[
            dates, ('open_price', 'high_price',
                    'low_price', 'close_price',
                    'volume',
                    'simple_roi')]

        print("[{}/{}] {} load to xarray OK, {:.3f} secs".format(
            sdx + 1, len(symbols), symbol, time() - t1))

    # # fill na with 0
    # pnl = pnl.fillna(0)

    # output data file path
    xarr.to_netcdf(fout_path)

    print("all exp_symbols load to xarray OK, {:.3f} secs".format(time() - t0))


def run_tej_csv_to_xarray(trim_start_date=dt.date(2000, 1, 3),
                          trim_end_date=dt.date(2018, 3, 15)
                          ):
    with open(pp.TAIEX_2005_LARGEST4ED_MARKET_CAP_SYMBOL_JSON) as fin:
        symbols = json.load(fin)

    csv_dir = os.path.join(pp.DATA_DIR, 'tej_csv')
    df_dir = pp.TMP_DIR

    # run
    tej_csv_to_df(symbols, csv_dir, df_dir)
    dataframe_to_xarray(symbols, df_dir, trim_start_date, trim_end_date,
                        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)


def symbol_statistics(start_date=dt.date(2005, 1, 1),
                      end_date=dt.date(2014, 12, 31)):
    """
    the statistics of the return of the specified stocks
    """
    import csv
    import json
    import statsmodels.tsa.stattools as tsa_tools
    import scipy.stats as spstats
    import portfolio_programming.statistics.risk_adjusted as risk_adj
    import arch.bootstrap.multiple_comparison as arch_comp

    symbols = json.load(open(os.path.join(pp.DATA_DIR,
                           'TAIEX_20050103_50largest_listed_market_cap.json')))
    data_xarr = xr.open_dataarray(os.path.join(pp.DATA_DIR,
                       'TAIEX_20050103_50largest_listed_market_cap_xarray.nc'))

    with open(os.path.join(pp.TMP_DIR,
       'TAIEX_20050103_50largest_listed_market_cap_stat.csv'), 'w') as csv_file:
        fields = ["rank", 'symbol', 'start_date', 'end_date', "n_data",
                  "cum_roi", "annual_roi", "roi_mu", "std", "skew", "ex_kurt",
                  "Sharpe", "Sortino", "JB", "worst_ADF", "SPA_c"]

        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for sdx, symbol in enumerate(symbols):
            rois = data_xarr.loc[start_date:end_date, symbol, 'simple_roi']
            trans_dates = rois.get_index('trans_date')
            n_roi = int(rois.count())
            rois[0] = 0
            cumulative_roi = float((1 + rois).prod() - 1)
            annual_roi = float(np.power(cumulative_roi + 1, 1. / 10) - 1)

            sharpe = risk_adj.Sharpe(rois)
            sortino = risk_adj.Sortino_full(rois)[0]
            jb = spstats.jarque_bera(rois)[1]

            # worse case of adf
            adf_c = tsa_tools.adfuller(rois, regression='c')[1]
            adf_ct = tsa_tools.adfuller(rois, regression='ct')[1]
            adf_ctt = tsa_tools.adfuller(rois, regression='ctt')[1]
            adf_nc = tsa_tools.adfuller(rois, regression='nc')[1]
            adf = max(adf_c, adf_ct, adf_ctt, adf_nc)

            spa_value = 0
            for _ in range(5):
                spa = arch_comp.SPA(rois.data, np.zeros(rois.size), reps=5000)
                spa.seed(np.random.randint(0, 2 ** 31 - 1))
                spa.compute()
                # preserve the worse p_value
                if spa.pvalues[1] > spa_value:
                    spa_value = spa.pvalues[1]

            writer.writerow({
                "rank": sdx + 1,
                "symbol": symbol,
                "start_date": trans_dates[0].strftime("%Y-%m-%d"),
                "end_date": trans_dates[-1].strftime("%Y-%m-%d"),
                "n_data": n_roi,
                "cum_roi": cumulative_roi,
                "annual_roi": annual_roi,
                "roi_mu": float(rois.mean()),
                "std": float(rois.std(ddof=1)),
                "skew": spstats.skew(rois, bias=False),
                "ex_kurt": spstats.kurtosis(rois, bias=False),
                "Sharpe": sharpe,
                "Sortino": sortino,
                "JB": jb,
                "worst_ADF": adf,
                "SPA_c": spa_value,
            })
            print("[{}/{}] {}, cum_roi:{:.2%}".format(
                sdx+1, len(symbols),
                symbol, cumulative_roi))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", default=False,
                        action='store_true',
                        help="csv to xarray")
    parser.add_argument("-s", '--stat', default=False,
                        action='store_true',
                        help="symbol statistics")

    args = parser.parse_args()
    if args.csv:
        run_tej_csv_to_xarray()
    elif args.stat:
        symbol_statistics()
