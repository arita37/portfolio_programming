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
                           'TAIEX_20050103_50largest_listed_market_cap_stat.csv'),
              'w') as csv_file:
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
                sdx + 1, len(symbols),
                symbol, cumulative_roi))


def plot_fft(symbol, start_date=dt.date(2005, 1, 1),
             end_date=dt.date(2017, 12, 31)):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates


    t0 = time()
    data_xarr = xr.open_dataarray(os.path.join(pp.DATA_DIR,
                       'TAIEX_20050103_50largest_listed_market_cap_xarray.nc'))

    ys = data_xarr.loc[start_date:end_date, symbol, 'simple_roi'] * 100
    n_point = int(ys.count())
    xs = ys.get_index('trans_date')

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    pts = np.arange(n_point)

    # freq domain
    freq_ys = np.fft.fft(ys)
    freq_ys = freq_ys[range(int(n_point / 2))]

    fig, ax = plt.subplots(2, 2)

    fig.suptitle('TAIEX {} {}-{}'.format(
        symbol,
        xs[0].strftime("%Y-%m-%d"), xs[-1].strftime("%Y-%m-%d")),
        fontsize=20)
    price_ax = ax[0, 0]
    hist_ax = ax[0, 1]
    roi_ax = ax[1, 0]
    fft_ax = ax[1, 1]

    datemin = np.datetime64(xs.date[0], 'Y')
    datemax = np.datetime64(xs.date[-1], 'Y') + np.timedelta64(1, 'Y')

    price_ax.plot(xs, data_xarr.loc[start_date:end_date, symbol,
                      'close_price'])
    price_ax.set_xlabel('Year', fontsize=12)
    price_ax.set_ylabel('Close price', fontsize=12)
    price_ax.xaxis.set_major_locator(years)
    price_ax.xaxis.set_major_formatter(yearsFmt)
    price_ax.xaxis.set_minor_locator(months)
    price_ax.set_xlim(datemin, datemax)
    price_ax.grid(True)

    hist_ax.hist(ys.data, bins=100, density=True, facecolor='green')
    hist_ax.set_xlabel('Return(%)', fontsize=12)
    hist_ax.set_ylabel('Probability density', fontsize=12)
    hist_ax.set_xlim(-10, 10)

    roi_ax.set_ylabel('Return(%)', fontsize=12)
    roi_ax.xaxis.set_major_locator(years)
    roi_ax.xaxis.set_major_formatter(yearsFmt)
    roi_ax.xaxis.set_minor_locator(months)
    roi_ax.plot(xs, ys)
    roi_ax.set_xlabel('Year', fontsize=12)
    roi_ax.set_xlim(datemin, datemax)
    roi_ax.grid(True)

    fft_ax.plot(abs(freq_ys), 'r')  # plotting the spectrum
    fft_ax.set_xlabel('Frequency (Hz)', fontsize=12)
    fft_ax.set_ylabel('|Y(freq)|', fontsize=12)
    fft_ax.set_xlim(0, n_point // 2)
    fft_ax.grid(True)

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    fig_path = os.path.join(pp.TMP_DIR, '{}_roi_{}_{}.png'.format(
        symbol, xs[0].strftime("%Y%m%d"), xs[-1].strftime("%Y%m%d")))
    fig.set_size_inches(16, 9)
    plt.savefig(fig_path, dpi=240, format='png')

    plt.show()
    print("symbol {} plot FFT complete, {:.4f} secs".format(symbol,
                                                            time()-t0))


def plot_hht(symbol, start_date=dt.date(2005, 1, 1),
             end_date=dt.date(2017, 12, 31)):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pyhht as hht

    t0 = time()
    data_xarr = xr.open_dataarray(os.path.join(pp.DATA_DIR,
                                               'TAIEX_20050103_50largest_listed_market_cap_xarray.nc'))

    ys = data_xarr.loc[start_date:end_date, symbol, 'simple_roi'] * 100
    n_point = int(ys.count())
    xs = ys.get_index('trans_date')
    datemin = np.datetime64(xs.date[0], 'Y')
    datemax = np.datetime64(xs.date[-1], 'Y') + np.timedelta64(1, 'Y')
    
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    # HHT
    decomposer = hht.EMD(ys.data)
    imfs = decomposer.decompose()
    n_imfs = imfs.shape[0]
    print('imfs shape:', imfs.shape)
    print(imfs[-1, :])
    print(imfs[n_imfs - 1, :])

    global_ylim = max(np.max(np.abs(imfs[:-1, :]), axis=0))

    plt.figure(figsize=(32, 18))
    hht_ax = plt.subplot(n_imfs + 1, 1, 1)
    hht_ax.plot(xs, ys.data)
    hht_ax.xaxis.set_major_locator(years)
    hht_ax.xaxis.set_major_formatter(yearsFmt)
    hht_ax.xaxis.set_minor_locator(months)
    hht_ax.set_xlim(datemin, datemax)
    hht_ax.set_ylim(ys.min(), ys.max())
    hht_ax.tick_params(axis='both', which='both', labelsize=8)
    # hht_ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
    #                labelbottom=False)
    hht_ax.grid(False)
    hht_ax.set_ylabel('Signal')
    hht_ax.set_title('Empirical Mode Decomposition')
    # Plot the IMFs
    for idx in range(n_imfs):
        imf_ax = plt.subplot(n_imfs + 1, 1, idx + 2)
        imf_ax.plot(xs, imfs[idx, :])
        imf_ax.xaxis.set_major_locator(years)
        imf_ax.xaxis.set_major_formatter(yearsFmt)
        imf_ax.xaxis.set_minor_locator(months)
        imf_ax.set_xlim(datemin, datemax)
        # imf_ax.set_ylim(-axis_extent, axis_extent)
        imf_ax.tick_params(axis='both', which='both', labelsize=8)
        imf_ax.grid(False)
        if idx != n_imfs - 1:
            imf_ax.set_ylabel('IMF' + str(idx + 1))
        else:
            imf_ax.set_ylabel('Residual')

    # res_ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    # res_ax.plot(xs, imfs[-1, :], 'r')
    plt.subplots_adjust(hspace=0.8)
    # hht.visualization.plot_imfs(ys.data, imfs)
    plt.show()

if __name__ == '__main__':
    # import json
    #
    # symbols = json.load(open(os.path.join(pp.DATA_DIR,
    #                             'TAIEX_20050103_50largest_listed_market_cap.json')))
    # for symbol in symbols:
    #     plot_fft(symbol)
    plot_hht("2412")

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
