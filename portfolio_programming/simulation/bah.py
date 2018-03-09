# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import sys
import logging
import os
import pickle
import platform
from time import time

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial)
from spsp_cvar import ValidMixin


class BAHPortfolio(ValidMixin):

    def __init__(self,
                 symbols,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=0.001425,
                 sell_trans_fee=0.004425,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 print_interval=10
                 ):
        """
        uniform buy-and-hold portfolio
        that is, one invests wealth among a pool of assets with an initial
        portfolio b1 and holds the portfolio until the end.

        Parameters:
        -------------
        symbols: list of sting,
            size: n_symbol

        risk_rois : xarray.DataArray,
            dim:(trans_date, symbol),
            shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        risk_free_rois : xarray.DataArray,
            dim: (trans_date),
            shape: (n_exp_period, )
            The return of risk-free asset, usually all zeros.

        initial_risk_wealth : xarray.DataArray, shape: (n_symbol,)
            The invested wealth of the stocks in the candidate set.

        initial_risk_free_wealth : float
            The initial wealth in the bank or the risky-free asset.

         buy_trans_fee : float
            The fee usually not change in the simulation.

        sell_trans_fee : float,
             The fee usually not change in the simulation.

        start_date : datetime.date
            The first trading date (not the calendar day) of simulation.

        end_date : datetime.date
             The last trading date (not the calendar day) of simulation.
        """
        self.symbols = symbols
        self.n_symbol = len(symbols)
        self.risk_free_symbol = 'risk_free'
        self.pf_symbols = symbols + [self.risk_free_symbol, ]

        # pandas.core.indexes.datetimes.DatetimeIndex
        self.all_trans_dates = risk_rois.get_index('trans_date')
        self.n_all_period = len(self.all_trans_dates)

        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # valid number of symbols
        self.valid_dimension("n_symbol", len(symbols),
                             len(initial_risk_wealth))
        self.initial_risk_wealth = initial_risk_wealth
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # verify transaction fee
        self.valid_range_value("buy_trans_fee", buy_trans_fee, 0, 1)
        self.buy_trans_fee = buy_trans_fee

        self.valid_range_value("sell_trans_fee", sell_trans_fee, 0, 1)
        self.sell_trans_fee = sell_trans_fee

        # .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[start_date:end_date]

        # date index in total data
        self.exp_trans_dates = self.exp_risk_rois.get_index('trans_date')
        self.n_exp_period = len(self.exp_trans_dates)
        self.exp_start_date = self.exp_trans_dates[0]
        self.exp_end_date = self.exp_trans_dates[self.n_exp_period - 1]

        self.exp_start_date_idx = self.all_trans_dates.get_loc(
            self.exp_start_date)
        self.exp_end_date_idx = self.all_trans_dates.get_loc(
            self.exp_end_date)

        self.valid_nonnegative_value("print_interval", print_interval)
        self.print_interval = print_interval

        # results data
        # decision xarray, shape: (n_exp_period, n_symbol+1, 3)
        decisions = ["wealth", "buy", "sell"]
        self.decision_xarr = xr.DataArray(
            np.zeros((self.n_exp_period,
                      self.n_symbol + 1,
                      len(decisions))),
            dims=('trans_date', 'symbol', 'decision'),
            coords=(
                self.exp_trans_dates,
                self.pf_symbols,
                decisions
            )
        )

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        func_name: str, Function name of the class
        """
        return "BAH_{}_{}_M{}".format(
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d"),
            self.n_symbol)

    @staticmethod
    def get_performance_report(
            simulation_name,
            symbols,
            risk_free_symbol,
            exp_start_date,
            exp_end_date,
            n_exp_period,
            buy_trans_fee,
            sell_trans_fee,
            initial_wealth,
            final_wealth,
            cum_trans_fee_loss,
            decision_xarr,
    ):
        """
       simulation reports

        Parameters:
        ------------------
        simulation_name : string
        symbols: list of string
            the candidate symbols in the simulation
        risk_free_symbol: string
        start_date, end_date: datetime.date
            the starting and ending days of the simulation
        buy_trans_fee, sell_trans_fee: float
            the transaction fee in the simulation
        initial_wealth, final_wealth: float
        n_exp_period: integer
        cum_trans_fee_loss: float
        decision xarray.DataArray, shape: (n_exp_period, n_stock+1, 5)
        """
        reports = dict()

        # basic information
        reports['os_uname'] = "|".join(platform.uname())
        reports['simulation_name'] = simulation_name
        reports['symbols'] = symbols
        reports['risk_free_symbol'] = risk_free_symbol
        reports['exp_start_date'] = exp_start_date
        reports['exp_end_date'] = exp_end_date
        reports['n_exp_period'] = n_exp_period
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['initial_wealth'] = initial_wealth
        reports['final_wealth'] = final_wealth
        reports['cum_trans_fee_loss'] = cum_trans_fee_loss
        reports['decision_xarr'] = decision_xarr

        # analysis
        reports['n_symbol'] = len(symbols)
        reports['cum_roi'] = final_wealth / initial_wealth - 1.
        reports['daily_roi'] = np.power(final_wealth / initial_wealth,
                                        1. / n_exp_period) - 1

        # wealth_arr, Pandas.Series, shape: (n_stock+1,)
        wealth_arr = decision_xarr.loc[:, :, 'wealth'].sum(axis=1).to_series()
        wealth_daily_rois = wealth_arr.pct_change()
        wealth_daily_rois[0] = 0

        reports['daily_mean_roi'] = wealth_daily_rois.mean()
        reports['daily_std_roi'] = wealth_daily_rois.std()
        reports['daily_skew_roi'] = wealth_daily_rois.skew()

        # excess Kurtosis
        reports['daily_ex-kurt_roi'] = wealth_daily_rois.kurt()
        reports['Sharpe'] = Sharpe(wealth_daily_rois)
        reports['Sortino_full'], reports['Sortino_full_semi_std'] = \
            Sortino_full(wealth_daily_rois)

        reports['Sortino_partial'], reports['Sortino_partial_semi_std'] = \
            Sortino_partial(wealth_daily_rois)

        return reports

    def run(self):
        """
        run simulation
        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get function name
        simulation_name = self.get_simulation_name()

        # current wealth of each stock in the portfolio
        cum_trans_fee_loss = 0

        # the first period, uniformly allocation money to each stock
        # the transaction fee  should be considered while buying

        self.decision_xarr.loc[self.exp_start_date, self.symbols, 'wealth'] = \
            np.ones(self.n_symbol) * \
            self.initial_risk_free_wealth / self.n_symbol / \
            (1 + self.buy_trans_fee)

        self.decision_xarr.loc[self.exp_start_date, self.symbols, 'buy'] = \
            self.decision_xarr.loc[self.exp_start_date, self.symbols, 'wealth']

        cum_trans_fee_loss += (np.ones(self.n_symbol) *
                               self.initial_risk_free_wealth *
                               self.buy_trans_fee /
                               self.n_symbol).sum()

        for tdx in range(1, self.n_exp_period):
            t1 = time()
            yesterday = self.exp_trans_dates[tdx - 1]
            today = self.exp_trans_dates[tdx]

            # no-action,only update wealth
            self.decision_xarr.loc[today, self.symbols, 'wealth'] = (
                    (1 + self.exp_risk_rois.loc[today, self.symbols]) *
                    self.decision_xarr.loc[yesterday, self.symbols,
                                           'wealth']
            )
            self.decision_xarr.loc[today, self.risk_free_symbol, 'wealth'] \
                = (
                    (1 + self.exp_risk_free_rois.loc[today]) *
                    self.decision_xarr.loc[yesterday, self.risk_free_symbol,
                                           'wealth']
            )
            if tdx % self.print_interval == 0:
                logging.info("{} [{}/{}] {} "
                             "wealth:{:.2f}, {:.3f} secs".format(
                    simulation_name,
                    tdx + 1,
                    self.n_exp_period,
                    today.strftime("%Y%m%d"),
                    float(self.decision_xarr.loc[today, :, 'wealth'].sum()),
                    time() - t1)
                )

        # sell at last day
        self.decision_xarr.loc[self.exp_end_date, self.symbols, 'sell'] = \
            self.decision_xarr.loc[self.exp_end_date, self.symbols, 'wealth']

        self.decision_xarr.loc[self.exp_end_date, self.risk_free_symbol,
                               'wealth'] = \
            (self.decision_xarr.loc[self.exp_end_date, self.symbols, 'wealth']
             * (1 - self.sell_trans_fee)).sum()

        cum_trans_fee_loss += (self.decision_xarr.loc[self.exp_end_date,
                                                      self.symbols, 'wealth'] *
                               self.sell_trans_fee).sum()
        # end of transaction

        # end of iterations, computing statistics
        initial_wealth = float(
                self.initial_risk_wealth.sum() + self.initial_risk_free_wealth)
        final_wealth = float(self.decision_xarr.loc[self.exp_end_date, :,
                       'wealth'].sum())
        # get reports
        reports = self.get_performance_report(
            simulation_name,
            self.symbols,
            self.risk_free_symbol,
            self.exp_start_date,
            self.exp_end_date,
            self.n_exp_period,
            self.buy_trans_fee,
            self.sell_trans_fee,
            float(initial_wealth),
            float(final_wealth),
            float(cum_trans_fee_loss),
            self.decision_xarr,
        )

        # add simulation time
        reports['simulation_time'] = time() - t0

        # write report
        bah_report_dir = os.path.join(pp.REPORT_DIR, 'bah')
        if not os.path.exists(bah_report_dir):
            os.makedirs(bah_report_dir)

        report_path = os.path.join(bah_report_dir,
                                   "report_{}.pkl".format(simulation_name))

        with open(report_path, 'wb') as fout:
            pickle.dump(reports, fout, pickle.HIGHEST_PROTOCOL)

        print("{}-{} {} OK, ROI:{:.2%} {:.4f} secs".format(
            platform.node(),
            os.getpid(),
            simulation_name,
            final_wealth/initial_wealth,
            time() - t0)
        )

        return reports


def run_bah(n_symbol):
    risky_roi_xarr = xr.open_dataarray(
        pp.TAIEX_2005_LARGESTED_MARKET_CAP_DATA_NC)
    symbols = list(risky_roi_xarr.get_index('symbol')[:n_symbol])
    exp_start_date = pp.EXP_START_DATE
    exp_end_date = pp.EXP_END_DATE

    risky_rois = risky_roi_xarr.loc[exp_start_date:exp_end_date,
                 symbols, 'simple_roi']
    exp_trans_dates = risky_rois.get_index('trans_date')
    n_exp_dates = len(exp_trans_dates)
    risk_free_rois = xr.DataArray(np.zeros(n_exp_dates),
                                  coords=(exp_trans_dates,))
    initial_risk_wealth = xr.DataArray(np.zeros(n_symbol),
                                       dims=('symbol',),
                                       coords=(symbols,))
    initial_risk_free_wealth = 1e6
    print(exp_start_date, exp_end_date, n_symbol)
    instance = BAHPortfolio(
        symbols,
        risky_rois,
        risk_free_rois,
        initial_risk_wealth,
        initial_risk_free_wealth,
        start_date=exp_start_date,
        end_date=exp_end_date,
    )
    instance.run()


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        format='%(filename)15s %(levelname)10s %(asctime)s\n'
               '%(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        level=logging.INFO)

    for m in range(1,51):
        run_bah(m)
