# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>

transforming TEJ's stock csv to xarray data
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
        df = pd.read_csv(
            open(csv_name),
            index_col="year_month_day",
            parse_dates=True,  # parse the index column
            dtype={
                # 'symbol': str,
                # 'abbreviation': str,
                "year_month_day": str,
                "open_price": np.float,
                "high_price": np.float,
                "low_price": np.float,
                "close_price": np.float,
                "volume_1000_shares": np.float,
                "value_1000_dollars": np.float,
                "simple_roi_%": np.float,
                "turnover_ratio_%": np.float,
                "market_value_million_dollars": np.float,
                "continuous_roi_%": np.float,
            },
            #converters={
                # "symbol": lambda x: x.strip(),
                # "abbreviation": lambda x: x.strip(),
            #},
        )

        # output data file path
        fout_path = os.path.join(df_dir, "{}_df.pkl".format(symbol))
        df.to_pickle(fout_path)

        print(
            "[{}/{}]{} csv to dataframe OK dates:{}_{}, {:.4f} secs".format(
                rdx + 1, len(symbols), symbol, df.index[0], df.index[-1],
                time() - t0)
        )

    print("tej csv_to_to OK, elapsed {:.4f} secs".format(time() - t0))


def yahoo_us_to_df(symbols, csv_dir, df_dir):
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
        print(csv_name)
        df = pd.read_csv(
            open(csv_name),
            index_col="Date",
            parse_dates=True,  # parse the index column
            dtype={
                "Date": str,
                "Open": np.float,
                "High": np.float,
                "Low": np.float,
                "Close": np.float,
                "Adj Close": np.float,
                "Volume": np.float,
            },
            converters={
            },
        )

        # rename index and columns
        df.index.name = "year_month_day"
        df.rename(
            columns={'Open': 'open_price',
                     'High': 'high_price',
                     "Low": "low_price",
                     'Close': 'close_price',
                     'Adj Close': "adj_close_price",
                     'Volume': "volume_shares"
                     },
            inplace=True)

        # simple roi
        df["simple_roi_%"] = df.loc[:, 'adj_close_price'].pct_change(
            fill_method=None) * 100

        # output data file path
        fout_path = os.path.join(df_dir, "{}_df.pkl".format(symbol))
        df.to_pickle(fout_path)

        print(
            "[{}/{}]{} csv to dataframe OK dates:{}_{}, {:.4f} secs".format(
                rdx + 1, len(symbols), symbol, df.index[0], df.index[-1],
                time() - t0
            )
        )

    print("US csv_to_to OK, elapsed {:.4f} secs".format(time() - t0))


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
    trans_dates.name = "trans_date"
    minor_indices = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "simple_roi",
    ]

    # setting xarray (date, symbol, indices)
    xarr = xr.DataArray(
        np.zeros((len(trans_dates), len(symbols), len(minor_indices))),
        dims=("trans_date", "symbol", "data"),
        coords=[trans_dates, symbols, minor_indices],
    )

    for sdx, symbol in enumerate(symbols):
        t1 = time()
        # read df
        fin_path = os.path.join(df_dir, "{}_df.pkl".format(symbol))
        trimmed_df = pd.read_pickle(fin_path).loc[start_date:end_date]

        # rename columns
        dates = trimmed_df.index
        trimmed_df["simple_roi_%"] /= 100.0
        trimmed_df.rename(columns={r"simple_roi_%": "simple_roi"},
                          inplace=True)

        trimmed_df.rename(columns={r"volume_1000_shares": "volume"},
                          inplace=True)

        xarr.loc[dates, symbol, :] = trimmed_df.loc[
            dates,
            (
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "simple_roi",
            ),
        ]

        print(
            "[{}/{}] {} load to xarray OK, {:.3f} secs".format(
                sdx + 1, len(symbols), symbol, time() - t1
            )
        )

    # # fill na with 0
    # pnl = pnl.fillna(0)

    # output data file path
    xarr.to_netcdf(fout_path)

    print("all exp_symbols load to xarray OK, {:.3f} secs".format(time() - t0))


def run_tej_csv_to_xarray(exp_name):
    if exp_name == "stocksp_cor15":
        with open(pp.TAIEX_2005_MKT_CAP_50_SYMBOL_JSON) as fin:
            tw_symbols = json.load(fin)

        csv_dir = os.path.join(pp.DATA_DIR, exp_name, "tej_csv")
        df_dir = pp.TMP_DIR
        trim_start_date = dt.date(2000, 1, 3)
        trim_end_date = dt.date(2018, 3, 15)

        # run
        tej_csv_to_df(tw_symbols, csv_dir, df_dir)
        dataframe_to_xarray(
            tw_symbols, df_dir, trim_start_date, trim_end_date,
            pp.TAIEX_2005_MKT_CAP_NC
        )

    elif exp_name == "dissertation":
        with open(pp.TAIEX_2005_MKT_CAP_50_SYMBOL_JSON) as tw_fin:
            tw_symbols = json.load(tw_fin)

        with open(pp.DJIA_2005_SYMBOL_JSON) as us_fin:
            djia_symbols = json.load(us_fin)

        tw_csv_dir = os.path.join(pp.DATA_DIR, "tej_csv")
        djia_csv_dir = os.path.join(pp.DATA_DIR, "yahoo_us_csv")
        df_dir = pp.TMP_DIR
        trim_start_date = dt.date(2000, 1, 3)
        trim_end_date = dt.date(2018, 12, 31)

        # run
        yahoo_us_to_df(djia_symbols, djia_csv_dir, df_dir)
        dataframe_to_xarray(
            djia_symbols, df_dir, trim_start_date, trim_end_date,
            pp.DJIA_2005_NC
        )
        tej_csv_to_df(tw_symbols, tw_csv_dir, df_dir)
        dataframe_to_xarray(
            tw_symbols, df_dir, trim_start_date, trim_end_date,
            pp.TAIEX_2005_MKT_CAP_NC
        )
    else:
        raise ValueError("unknown exp_name:{}".format(exp_name))


def symbol_statistics(exp_name):
    """
    the statistics of the return of the specified stocks
    """
    import csv
    import json
    import statsmodels.tsa.stattools as tsa_tools
    import scipy.stats as spstats
    import portfolio_programming.statistics.risk_adjusted as risk_adj
    import arch.bootstrap.multiple_comparison as arch_comp

    if exp_name == 'stocksp_cor15':
        start_date = dt.date(2005, 1, 1)
        end_date = dt.date(2014, 12, 31)

        with open(pp.TAIEX_2005_MKT_CAP_50_SYMBOL_JSON) as fin:
            symbols = json.load(fin)

        data_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)

        with open(
                os.path.join(pp.TMP_DIR,
                    "TAIEX_20050103_50largest_listed_market_cap_stat.csv"),
                "w",
        ) as csv_file:
            fields = [
                "rank",
                "symbol",
                "start_date",
                "end_date",
                "n_data",
                "cum_roi",
                "annual_roi",
                "roi_mu",
                "std",
                "skew",
                "ex_kurt",
                "Sharpe",
                "Sortino",
                "JB",
                "worst_ADF",
                "SPA_c",
            ]

            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for sdx, symbol in enumerate(symbols):
                rois = data_xarr.loc[start_date:end_date, symbol, "simple_roi"]
                trans_dates = rois.get_index("trans_date")
                n_roi = int(rois.count())
                rois[0] = 0
                cumulative_roi = float((1 + rois).prod() - 1)
                annual_roi = float(np.power(cumulative_roi + 1, 1.0 / 10) - 1)

                sharpe = risk_adj.Sharpe(rois)
                sortino = risk_adj.Sortino_full(rois)[0]
                jb = spstats.jarque_bera(rois)[1]

                # worse case of adf
                adf_c = tsa_tools.adfuller(rois, regression="c")[1]
                adf_ct = tsa_tools.adfuller(rois, regression="ct")[1]
                adf_ctt = tsa_tools.adfuller(rois, regression="ctt")[1]
                adf_nc = tsa_tools.adfuller(rois, regression="nc")[1]
                adf = max(adf_c, adf_ct, adf_ctt, adf_nc)

                spa_value = 0
                for _ in range(5):
                    spa = arch_comp.SPA(rois.data, np.zeros(rois.size),
                                        reps=5000)
                    spa.seed(np.random.randint(0, 2 ** 31 - 1))
                    spa.compute()
                    # preserve the worse p_value
                    if spa.pvalues[1] > spa_value:
                        spa_value = spa.pvalues[1]

                writer.writerow(
                    {
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
                    }
                )
                print(
                    "[{}/{}] {}, cum_roi:{:.2%}".format(
                        sdx + 1, len(symbols), symbol, cumulative_roi
                    )
                )
    elif exp_name == 'dissertation':
        start_date = dt.date(2005, 1, 1)
        end_date = dt.date(2018, 12, 31)

        with open(pp.DJIA_2005_SYMBOL_JSON) as us_fin:
            djia_symbols = json.load(us_fin)

        djia_xarr = xr.open_dataarray(pp.DJIA_2005_NC)
        djia_stats_file = os.path.join(pp.TMP_DIR, "DJIA_2005_symbols_stat.csv")

        with open(pp.TAIEX_2005_MKT_CAP_50_SYMBOL_JSON) as tw_fin:
            tw_symbols = json.load(tw_fin)

        tw_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
        tw_stats_file = os.path.join(pp.TMP_DIR,
                                     "TAIEX_2005_market_cap_stat.csv")

        for mkt, symbols, data_xarr, stat_file in zip(['djia', 'tw'],
                                           [djia_symbols, tw_symbols],
                                           [djia_xarr, tw_xarr],
                                           [djia_stats_file, tw_stats_file]):

            with open(stat_file, "w",) as csv_file:
                fields = [
                    "rank",
                    "symbol",
                    "start_date",
                    "end_date",
                    "n_data",
                    "cum_roi",
                    "annual_roi",
                    "roi_mu",
                    "std",
                    "skew",
                    "ex_kurt",
                    "Sharpe",
                    "Sortino",
                    "JB",
                    "worst_ADF",
                    "SPA_c",
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()

                for sdx, symbol in enumerate(symbols):
                    rois = data_xarr.loc[start_date:end_date, symbol,
                           "simple_roi"]
                    trans_dates = rois.get_index("trans_date")
                    n_roi = int(rois.count())
                    rois[0] = 0
                    cumulative_roi = float((1 + rois).prod() - 1)
                    annual_roi = float(np.power(cumulative_roi + 1, 1.0 / 14)
                                       - 1)

                    sharpe = risk_adj.Sharpe(rois)
                    sortino = risk_adj.Sortino_full(rois)[0]
                    jb = spstats.jarque_bera(rois)[1]

                    # worse case of adf
                    adf_c = tsa_tools.adfuller(rois, regression="c")[1]
                    adf_ct = tsa_tools.adfuller(rois, regression="ct")[1]
                    adf_ctt = tsa_tools.adfuller(rois, regression="ctt")[1]
                    adf_nc = tsa_tools.adfuller(rois, regression="nc")[1]
                    adf = max(adf_c, adf_ct, adf_ctt, adf_nc)

                    spa_value = 0
                    for _ in range(5):
                        spa = arch_comp.SPA(rois.data, np.zeros(rois.size),
                                            reps=5000)
                        spa.seed(np.random.randint(0, 2 ** 31 - 1))
                        spa.compute()
                        # preserve the worse p_value
                        if spa.pvalues[1] > spa_value:
                            spa_value = spa.pvalues[1]

                    writer.writerow(
                        {
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
                        }
                    )
                    print(
                        "[{}/{}] {}, cum_roi:{:.2%}".format(
                            sdx + 1, len(symbols), symbol, cumulative_roi
                        )
                    )

    else:
        raise ValueError("unknown exp_name:{}".format(exp_name))


def plot_fft(symbol, start_date=dt.date(2005, 1, 1),
             end_date=dt.date(2017, 12, 31)):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    t0 = time()
    data_xarr = xr.open_dataarray(
        os.path.join(
            pp.DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap_xarray.nc"
        )
    )

    ys = data_xarr.loc[start_date:end_date, symbol, "simple_roi"] * 100
    n_point = int(ys.count())
    xs = ys.get_index("trans_date")

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter("%Y")

    pts = np.arange(n_point)

    # freq domain
    freq_ys = np.fft.fft(ys)
    freq_ys = freq_ys[range(int(n_point / 2))]

    fig, ax = plt.subplots(2, 2)

    fig.suptitle(
        "TAIEX {} {}-{}".format(
            symbol, xs[0].strftime("%Y-%m-%d"), xs[-1].strftime("%Y-%m-%d")
        ),
        fontsize=20,
    )
    price_ax = ax[0, 0]
    hist_ax = ax[0, 1]
    roi_ax = ax[1, 0]
    fft_ax = ax[1, 1]

    datemin = np.datetime64(xs.date[0], "Y")
    datemax = np.datetime64(xs.date[-1], "Y") + np.timedelta64(1, "Y")

    price_ax.plot(xs, data_xarr.loc[start_date:end_date, symbol, "close_price"])
    price_ax.set_xlabel("Year", fontsize=12)
    price_ax.set_ylabel("Close price", fontsize=12)
    price_ax.xaxis.set_major_locator(years)
    price_ax.xaxis.set_major_formatter(yearsFmt)
    price_ax.xaxis.set_minor_locator(months)
    price_ax.set_xlim(datemin, datemax)
    price_ax.grid(True)

    hist_ax.hist(ys.data, bins=100, density=True, facecolor="green")
    hist_ax.set_xlabel("Return(%)", fontsize=12)
    hist_ax.set_ylabel("Probability density", fontsize=12)
    hist_ax.set_xlim(-10, 10)

    roi_ax.set_ylabel("Return(%)", fontsize=12)
    roi_ax.xaxis.set_major_locator(years)
    roi_ax.xaxis.set_major_formatter(yearsFmt)
    roi_ax.xaxis.set_minor_locator(months)
    roi_ax.plot(xs, ys)
    roi_ax.set_xlabel("Year", fontsize=12)
    roi_ax.set_xlim(datemin, datemax)
    roi_ax.grid(True)

    fft_ax.plot(abs(freq_ys), "r")  # plotting the spectrum
    fft_ax.set_xlabel("Frequency (Hz)", fontsize=12)
    fft_ax.set_ylabel("|Y(freq)|", fontsize=12)
    fft_ax.set_xlim(0, n_point // 2)
    fft_ax.grid(True)

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    fig_path = os.path.join(
        pp.TMP_DIR,
        "{}_roi_{}_{}.png".format(
            symbol, xs[0].strftime("%Y%m%d"), xs[-1].strftime("%Y%m%d")
        ),
    )
    fig.set_size_inches(16, 9)
    plt.savefig(fig_path, dpi=240, format="png")

    plt.show()
    print("symbol {} plot FFT complete, {:.4f} secs".format(symbol, time() - t0))


def plot_hht(symbol, start_date=dt.date(2005, 1, 1),
             end_date=dt.date(2017, 12, 31)):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import PyEMD

    t0 = time()
    data_xarr = xr.open_dataarray(
        os.path.join(
            pp.DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap_xarray.nc"
        )
    )

    ys = data_xarr.loc[start_date:end_date, symbol, "simple_roi"] * 100
    xs = ys.get_index("trans_date")
    datemin = np.datetime64(xs.date[0], "Y")
    datemax = np.datetime64(xs.date[-1], "Y") + np.timedelta64(1, "Y")

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter("%Y")

    # HHT
    emds = {
        # "hht_emd": pyhht.EMD,
        "pyemd_emd": PyEMD.EMD,
        # "pyemd_eemd": PyEMD.EEMD,
        # "pyemd_eemdan": PyEMD.CEEMDAN
    }
    print("start HHT")
    for emd_name, emd in emds.items():
        t0 = time()
        if emd_name == "hht_emd":
            obj = emd(ys.data)
            imfs = obj.decompose()
        else:
            obj = emd()
            imfs = obj(ys.data)

        n_imfs = imfs.shape[0]
        print("{} {}".format(emd_name, n_imfs))
        global_ylim = max(np.max(np.abs(imfs[:-1, :]), axis=0))

        fig = plt.figure(figsize=(32, 18))
        hht_ax = plt.subplot(n_imfs + 1, 1, 1)
        hht_ax.plot(xs, ys.data)
        hht_ax.xaxis.set_major_locator(years)
        hht_ax.xaxis.set_major_formatter(yearsFmt)
        hht_ax.xaxis.set_minor_locator(months)
        hht_ax.set_xlim(datemin, datemax)
        hht_ax.set_ylim(ys.min(), ys.max())
        hht_ax.tick_params(axis="both", which="both", labelsize=8)
        # hht_ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
        #                labelbottom=False)
        hht_ax.grid(False)
        hht_ax.set_ylabel("Signal")
        hht_ax.set_title("Empirical Mode Decomposition")
        # Plot the IMFs
        for idx in range(n_imfs):
            imf_ax = plt.subplot(n_imfs + 1, 1, idx + 2)
            imf_ax.plot(xs, imfs[idx, :])
            imf_ax.xaxis.set_major_locator(years)
            imf_ax.xaxis.set_major_formatter(yearsFmt)
            imf_ax.xaxis.set_minor_locator(months)
            imf_ax.set_xlim(datemin, datemax)
            # imf_ax.set_ylim(-axis_extent, axis_extent)
            imf_ax.tick_params(axis="both", which="both", labelsize=8)
            imf_ax.grid(False)
            if idx != n_imfs - 1:
                imf_ax.set_ylabel("IMF" + str(idx + 1))
            else:
                imf_ax.set_ylabel("Residual")

        fig_path = os.path.join(
            pp.TMP_DIR,
            "{}_roi_{}_{}_{}.png".format(
                symbol, xs[0].strftime("%Y%m%d"), xs[-1].strftime("%Y%m%d"),
                emd_name
            ),
        )
        fig.set_size_inches(16, 9)
        plt.savefig(fig_path, dpi=240, format="png")
        print("save figure:{} OK, {:.4f} secs".format(fig_path, time() - t0))
        plt.subplots_adjust(hspace=0.8)

    # plt.show()


def plot_wavelet(
        symbol, start_date=dt.date(2017, 1, 1), end_date=dt.date(2017, 12, 31)
):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pywt

    t0 = time()
    data_xarr = xr.open_dataarray(
        os.path.join(
            pp.DATA_DIR, "TAIEX_20050103_50largest_listed_market_cap_xarray.nc"
        )
    )

    ys = data_xarr.loc[start_date:end_date, symbol, "simple_roi"] * 100
    xs = ys.get_index("trans_date")
    datemin = np.datetime64(xs.date[0], "Y")
    datemax = np.datetime64(xs.date[-1], "Y") + np.timedelta64(1, "Y")

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter("%Y")

    print(pywt.wavelist())
    waves = {"db20"}
    # 'sym20', 'coif5'
    # }
    for wave in waves:
        if wave in ("mexh", "morl"):
            w = pywt.ContinuousWavelet(wave)
        else:
            w = pywt.Wavelet(wave)
        a = ys.data
        ca = []
        cd = []
        for idx in range(10):
            (a, d) = pywt.dwt(a, w)
            ca.append(a)
            cd.append(d)

        rec_a = []
        rec_d = []

        for idx, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * idx
            rec_a.append(pywt.waverec(coeff_list, w))

        for idx, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * idx
            rec_d.append(pywt.waverec(coeff_list, w))

        fig = plt.figure()
        ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
        ax_main.set_title("{} wave:{}".format(symbol, w.name))
        ax_main.plot(ys.data)
        ax_main.set_xlim(0, len(ys) - 1)

        for idx, y in enumerate(rec_a):
            ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + idx * 2)
            ax.plot(y, "r")
            ax.set_xlim(0, len(y) - 1)
            ax.set_ylabel("A%d" % (idx + 1))

        for idx, y in enumerate(rec_d):
            ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + idx * 2)
            ax.plot(y, "g")
            ax.set_xlim(0, len(y) - 1)
            ax.set_ylabel("D%d" % (idx + 1))

    plt.show()


def run_plot_group_line_chart():
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = (['Times New Roman'] +
                                  plt.rcParams['font.serif'])

    start_date = dt.date(2005, 1, 1)
    end_date = dt.date(2018, 12, 31)

    with open(pp.TAIEX_2005_MKT_CAP_50_SYMBOL_JSON) as tw_fin:
        tw_symbols = json.load(tw_fin)

    tw_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)

    with open(pp.DJIA_2005_SYMBOL_JSON) as us_fin:
        djia_symbols = json.load(us_fin)

    djia_xarr = xr.open_dataarray(pp.DJIA_2005_NC)

    # A3: 11.3 * 17 inch
    mkt_fig, mkt_axes= plt.subplots(figsize=(11.3, 17), ncols=1, nrows=2)
    taiex_df = tw_xarr.loc[start_date:end_date, 'TAIEX',
               ['close_price', 'simple_roi']].to_pandas()
    djia_df = djia_xarr.loc[start_date:end_date, 'DJIA',
              ['close_price', 'simple_roi']].to_pandas()
    mkt_df = taiex_df.merge(djia_df, how="left", on='trans_date',
                            suffixes=('_TAIEX', '_DJIA'))
    mkt_df.loc[:, ['close_price_TAIEX','close_price_DJIA']].plot.line(
        ax=mkt_axes[0], grid=True, style=['-', '--'], fontsize=14)
    mkt_axes[0].set_title('Market index', fontsize=20)
    mkt_axes[0].set_xlabel('', fontsize=14)
    mkt_axes[0].set_ylabel("Close price", fontsize=16)
    # mkt_axes[0].set_xticklabels(range(2005, 2019+1))
    mkt_axes[0].set_xlim(dt.date(2004,12,20), dt.date(2019,1,10))
    mkt_axes[0].legend(['TAIEX', 'DJIA'], loc='lower right',
                     ncol=2, fancybox=True, shadow=True, fontsize=13)

    rois_df = mkt_df.loc[:, ['simple_roi_TAIEX', 'simple_roi_DJIA']]
    interval_grouper = (rois_df + 1).groupby(pd.Grouper(freq="M"))
    mon_df = (interval_grouper.prod() - 1) * 100

    mon_df.plot.line(
        ax=mkt_axes[1], grid=True, style=['-', '--'], fontsize=14)

    mkt_axes[1].set_title('', fontsize=20)
    mkt_axes[1].set_xlabel('', fontsize=14)
    mkt_axes[1].set_ylabel("Monthly return(%)", fontsize=16)
    mkt_axes[1].set_xlim(dt.date(2004,12,31), dt.date(2019,1,1))
    mkt_axes[1].legend(['TAIEX', 'DJIA'], loc='lower right',
                     ncol=2, fancybox=True, shadow=True, fontsize=13)
    mkt_img_file = os.path.join(pp.TMP_DIR, 'market_chart.pdf')
    plt.savefig(mkt_img_file, format='pdf')

    for mkt, xarr, symbols in zip(['US', 'TW'], [djia_xarr, tw_xarr],
                                  [djia_symbols, tw_symbols]):

        for fdx in range(2):
            fig, axes = plt.subplots(figsize=(11.3, 17), ncols=1, nrows=3)

            for gdx, sdx in enumerate(range(0, 15, 5)):

                df = xarr.loc[start_date: end_date,
                              symbols[fdx*15+sdx:fdx*15+sdx+5],
                             'simple_roi'].to_pandas()
                # frequency,  'M': calendar month end, 'Q'	calendar quarter end
                # 'A': calendar year end

                interval_grouper = (df+1).groupby(pd.Grouper(freq="M"))
                mon_df = (interval_grouper.prod()-1) * 100

                # axes[gdx].set_yticks(range(-60, 80, 10))

                mon_df.plot.line(ax=axes[gdx], grid=True,
                                 style=['-', '--', '-.', ':', '-'],
                                 fontsize=14)
                axes[gdx].set_title('{}_G{}'.format(mkt, fdx*3+gdx + 1),
                                    fontsize=20)
                axes[gdx].set_xlabel('', fontsize=14)
                axes[gdx].set_ylabel("Monthly return(%)", fontsize=16)
                axes[gdx].legend(loc='lower right',
                          ncol=5, fancybox=True, shadow=True, fontsize=13)

            img_file = os.path.join(pp.TMP_DIR,
                                    '{}_monthly_roi_chart_{}.pdf'.format(
                                        mkt, fdx+1))
            plt.savefig(img_file, format='pdf')

    #
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--csv", default=False, action="store_true",
        help="csv to xarray"
    )
    parser.add_argument(
        "-s", "--stat", default=False, action="store_true",
        help="symbol statistics"
    )
    parser.add_argument(
        "-p", "--plot",  default=False, action="store_true"
    )

    args = parser.parse_args()
    print("current experiment name: {}".format(pp.EXP_NAME))
    if args.csv:
        run_tej_csv_to_xarray(pp.EXP_NAME)
    elif args.stat:
        symbol_statistics(pp.EXP_NAME)
    elif args.plot:
        run_plot_group_line_chart()
