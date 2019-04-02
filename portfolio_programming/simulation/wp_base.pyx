# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>

basic trading flow schema for online portfolio
"""

import platform
from time import time
import numpy as np
import scipy.optimize as spopt
import xarray as xr
import os
import logging
import pickle

cimport numpy as cnp
import portfolio_programming as pp
from portfolio_programming.simulation.spsp_base import ValidMixin
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial)


def func_rebalance_opt(
        double today_portfolio_wealth,
        cnp.ndarray[cnp.float64_t, ndim=1] prev_weights,
        double prev_portfolio_wealth,
        cnp.ndarray[cnp.float64_t, ndim=1] price_relatives,
        cnp.ndarray[cnp.float64_t, ndim=1] today_weights,
        double buy_trans_fee,
        double sell_trans_fee):
    """
    The decision variable must be located in first place.

    Parameters:
    ------------------------
    today_portfolio_wealth: float,
        the portfolio wealth after rebalance, and it is the decision variable

    prev_weights: numpy.array like
        the stocks' weights of yesterday

    prev_portfolio_wealth, : float
        the portfolio wealth of yesterday

    price_relatives : numpy.array like,

    today_weights:   numpy.array like,
          the stocks' weights of today

    buy: float,
        buy transaction fee

    c_sell: float,
        sell transaction fee

    Returns:
    --------------------
    balance equation
    """

    today_prev_wealths = (prev_portfolio_wealth * prev_weights *
                          price_relatives)

    today_wealths = today_portfolio_wealth * today_weights
    buy_fee = buy_trans_fee * (
        np.maximum(today_wealths - today_prev_wealths, 0).sum()
    )
    sell_fee = sell_trans_fee * (
        np.maximum(today_prev_wealths - today_wealths, 0).sum()
    )

    balance = (today_portfolio_wealth - today_prev_wealths.sum() +
               buy_fee + sell_fee)
    return balance


class WeightPortfolio(ValidMixin):
    def __init__(self,
                 str group_name,
                 list symbols,
                 risk_rois,
                 initial_weights,
                 double initial_wealth=1e6,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 int print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        """
        allocating capital according to weights

        Parameters:
        -------------
        symbols: List[str]
            list of symbols

        group_name: string
            the portfolio's name

        risk_rois: xarray.DataArray,
            dim:(trans_date, symbol),
            shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        initial_weights: xarray.DataArray
            dim: (symbol, weights)

        initial_wealth: float
        buy_trans_fee: float
        sell_trans_fee: float
        start_date: datetime.date
        end_date: datetime.date

        print_interval : positive integer

        Data
        --------------

        """

        group_symbols = pp.GROUP_SYMBOLS

        if group_name not in group_symbols.keys():
            raise ValueError('Unknown group name:{}'.format(group_name))
        self.group_name = group_name

        self.symbols = symbols
        self.n_symbol = len(symbols)

        # truncate rois to experiment interval
        self.all_trans_dates = risk_rois.get_index('trans_date')
        self.n_all_period = len(self.all_trans_dates)

        self.exp_rois = risk_rois.loc[start_date:end_date]
        self.exp_trans_dates = self.exp_rois.get_index('trans_date')
        self.n_exp_period = len(self.exp_trans_dates)

        self.exp_start_date = self.exp_trans_dates[0]
        self.exp_end_date = self.exp_trans_dates[self.n_exp_period - 1]
        self.exp_start_date_idx = self.all_trans_dates.get_loc(
            self.exp_start_date)
        self.exp_end_date_idx = self.all_trans_dates.get_loc(
            self.exp_end_date)

        self.valid_dimension('n_symbol', len(symbols), len(initial_weights))
        self.initial_weights = initial_weights

        self.valid_positive_value('initial_wealth', initial_wealth)
        self.initial_wealth = initial_wealth

        self.valid_range_value("buy_trans_fee", buy_trans_fee, 0, 1)
        self.buy_trans_fee = buy_trans_fee

        self.valid_range_value("sell_trans_fee", sell_trans_fee, 0, 1)
        self.sell_trans_fee = sell_trans_fee

        self.valid_nonnegative_value("print_interval", print_interval)
        self.print_interval = print_interval

        # results data
        # decision xarray, shape: (n_exp_period, n_symbol, decisions)
        decisions = ["wealth", "weight", 'portfolio_payoff']
        self.decision_xarr = xr.DataArray(
            np.zeros((self.n_exp_period,
                      self.n_symbol,
                      len(decisions))
                     ),
            dims=('trans_date', 'symbol', 'decision'),
            coords=(
                self.exp_trans_dates,
                self.symbols,
                decisions
            )
        )

        # report path
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        self.report_path =  os.path.join(
            report_dir, "report_{}.pkl".format(self.get_simulation_name())
         )

    def get_simulation_name(self, *args, **kwargs):
        """implemented by user"""
        raise NotImplementedError('get_simulation_name() '
                                  'does not be implemented.')

    def get_today_weights(self, *args, **kwargs):
        """ implemented by user """
        raise NotImplementedError('get_current_weights() '
                                  'does not be implemented.')


    def add_to_reports(self, reports):
        """
        add specific results to reports after simulation
        """
        return reports

    def pre_trading_operation(self, *args, **kargs):
        """
        operations after initialization and before trading
        """

    def func_rebalance(self, current_portfolio_wealth,
                       prev_weights, prev_portfolio_wealth,
                       price_relatives, today_weights):

        """
        Parameters:
        -------------
        current_portfolio_wealth : float
            initial guess of the portfolio wealth after balanced

        prev_weights: numpy.array like
            the stocks' weights of yesterday

        prev_portfolio_wealth, : float
            the portfolio wealth of yesterday

        price_relatives : numpy.array like,

        today_weights:  numpy.array like,

        buy_trans_fee: float,
            buy transaction fee

        sell_trans_fee: float,
            sell transaction fee

        Returns:
        -------------
        today_portfolio_wealth: float,
            the portfolio wealth after rebalance
        """
        sol = spopt.newton(func_rebalance_opt,
                           current_portfolio_wealth,
                           args=(prev_weights,
                                 prev_portfolio_wealth,
                                 price_relatives,
                                 today_weights,
                                 self.buy_trans_fee,
                                 self.sell_trans_fee
                                 )
                           )
        return sol

    @staticmethod
    def get_performance_report(
            simulation_name,
            group_name,
            symbols,
            initial_weights,
            initial_wealth,
            buy_trans_fee,
            sell_trans_fee,
            exp_start_date,
            exp_end_date,
            n_exp_period,
            final_wealth,
            cum_trans_fee_loss,
            decision_xarr
    ):
        reports = dict()
        # basic information
        reports['os_uname'] = "|".join(platform.uname())
        reports['simulation_name'] = simulation_name
        reports['group_name'] = group_name
        reports['symbols'] = symbols
        reports['initial_weights'] = initial_weights
        reports['initial_wealth'] = initial_wealth
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['exp_start_date'] = exp_start_date
        reports['exp_end_date'] = exp_end_date
        reports['n_exp_period'] = n_exp_period
        reports['final_wealth'] = float(final_wealth)
        reports['cum_trans_fee_loss'] = float(cum_trans_fee_loss)
        reports['decision_xarr'] = decision_xarr

        # ROI analysis
        price_rel = float(final_wealth / initial_wealth)
        reports['cum_roi'] = (price_rel - 1)
        reports['daily_roi'] = np.power(price_rel,
                                        1. / n_exp_period) - 1
        year_interval = (exp_end_date.year - exp_start_date.year) + 1
        reports['annual_roi'] = np.power(price_rel, 1. / year_interval) - 1

        # risk analysis
        # 'trans_date', 'symbol', 'decision'
        wealths = decision_xarr.loc[:, :, 'wealth'].sum(axis=1).to_series()
        wealth_daily_rois = wealths.pct_change()
        wealth_daily_rois[0] = 0

        reports['daily_mean_roi'] = wealth_daily_rois.mean()
        reports['daily_std_roi'] = wealth_daily_rois.std()
        reports['daily_skew_roi'] = wealth_daily_rois.skew()
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

        prev_weights: array like, w_{t-1}
        prev_portfolio_wealth: float, w_{(p), t-1}
        today price relaitves: array like, x_t
        today_prev_wealth: array like, w_{(p), t-1} * w_{t-1} * x_t
        today_weights (rebalanced): w_t
        """
        t0 = time()
        simulation_name = self.get_simulation_name()
        cum_trans_fee_loss = 0

        # first allocation should also consider the transaction fee
        self.decision_xarr.loc[self.exp_start_date, self.symbols, 'weight'] = \
            self.initial_weights
        self.decision_xarr.loc[self.exp_start_date, self.symbols, 'wealth'] = (
                self.initial_wealth * self.initial_weights *
                (1-self.buy_trans_fee))
        # record portfolio payoff
        self.decision_xarr.loc[self.exp_start_date, self.symbols,
                               'portfolio_payoff'] = self.initial_weights

        cum_trans_fee_loss += (self.initial_wealth * self.buy_trans_fee)

        self.pre_trading_operation()

        # start trading
        for tdx in range(1, self.n_exp_period):
            t1 = time()
            yesterday = self.exp_trans_dates[tdx - 1]
            today = self.exp_trans_dates[tdx]

            # the cumulative wealth before rebalance
            # Note that we have already known today's ROIs
            today_price_relatives = (self.exp_rois.loc[today, self.symbols]
                                     + 1)
            today_prev_wealth = (today_price_relatives *
                     self.decision_xarr.loc[yesterday,
                                            self.symbols, 'wealth'])

            # record portfolio payoff
            self.decision_xarr.loc[today, self.symbols, 'portfolio_payoff'] = (
                 self.decision_xarr.loc[yesterday, self.symbols, 'weight'] *
                 today_price_relatives
            )

            today_prev_portfolio_wealth = today_prev_wealth.sum()
            # current_weights = current_wealth / current_total_wealth
            # print(today, self.exp_rois.loc[today, :]*100)
            # print(today, today_prev_wealth)

            # get today weights
            self.decision_xarr.loc[today, self.symbols, 'weight'] = (
                self.get_today_weights(
                    tdx=tdx,
                    prev_trans_date=yesterday,
                    trans_date=today,
                    today_price_relative=today_price_relatives,
                    today_prev_wealth=today_prev_wealth,
                    today_prev_portfolio_wealth=today_prev_portfolio_wealth,
                )
            )

            # the cumulative wealth after rebalance
            today_portfolio_wealth = self.func_rebalance(
                # initial guess
                today_prev_portfolio_wealth,
                # prev weights and portfolio wealth
                self.decision_xarr.loc[yesterday, self.symbols, 'weight'].values,
                self.decision_xarr.loc[yesterday, self.symbols, 'wealth'].sum(),
                # price relatives and weights of today
                today_price_relatives.values,
                self.decision_xarr.loc[today, self.symbols, 'weight'].values
            )

            cum_trans_fee_loss += (today_portfolio_wealth -
                                   today_prev_portfolio_wealth)

            # rebalance the wealth by CRP weights
            self.decision_xarr.loc[today, self.symbols, 'wealth'] = (
                    today_portfolio_wealth *
                    self.decision_xarr.loc[today,  self.symbols, 'weight']
            )

            if tdx % self.print_interval == 0:
                logging.info("{} [{}/{}] {} "
                             "wealth:{:.2f}, {:.3f} secs".format(
                    simulation_name,
                    tdx + 1,
                    self.n_exp_period,
                    today.strftime("%Y%m%d"),
                    float(self.decision_xarr.loc[today, self.symbols,
                                                 'wealth'].sum()),
                    time() - t1
                )
                )
        # end of loop

        final_wealth = self.decision_xarr.loc[
                       self.exp_end_date, self.symbols, 'wealth'].sum()

        reports = self.get_performance_report(
            simulation_name,
            self.group_name,
            self.symbols,
            self.initial_weights,
            self.initial_wealth,
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.exp_start_date,
            self.exp_end_date,
            self.n_exp_period,
            final_wealth,
            cum_trans_fee_loss,
            self.decision_xarr
        )

        # additional results to report
        reports['simulation_time'] = time() - t0
        reports = self.add_to_reports(reports)

        # write report
        with open(self.report_path, 'wb') as fout:
            pickle.dump(reports, fout, pickle.HIGHEST_PROTOCOL)

        print("{}-{} {} OK, {:.4f} secs".format(
            platform.node(),
            os.getpid(),
            simulation_name,
            time() - t0)
        )
        return reports

class NIRUtility(object):
    """
    internal regret general method
    """
    @staticmethod
    def modified_probabilities(probs):
        """
        Parameters:
        ------------
        probs: array like
            shape: n_action

        Returns:
        -------------------
        size * (size - 1) * n_action modified probabilities
        """
        n_action = len(probs)
        virtual_experts = np.tile(probs, (n_action * (n_action - 1), 1))
        row = 0
        for idx in range(n_action):
            for jdx in range(n_action):
                if idx != jdx:
                    virtual_experts[row, jdx] += virtual_experts[row, idx]
                    virtual_experts[row, idx] = 0
                    row += 1
        return virtual_experts


    @staticmethod
    def column_stochastic_matrix(n_action, virtual_expert_weights):
        """
        Parameters:
        ------------
        n_action: int, number of actions
        virtual_expert_weights: array like,
            shape: n_action * (n_action-1)

        Returns:
        -------------
        numpy.array, shape: n_action * n_action
        """
        n_virtual_expert = len(virtual_expert_weights)
        assert n_virtual_expert == n_action * (n_action - 1)

        # build row-sum-zero matrix
        A = np.insert(virtual_expert_weights,
                      np.arange(0, n_action * n_action, n_action),
                      np.zeros(n_action)).reshape((n_action, n_action))
        np.fill_diagonal(A, -A.sum(axis=1))

        # column stochastic matrix
        S = A.T / np.max(np.abs(A)) + np.identity(n_action)
        return S