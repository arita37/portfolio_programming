# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False

"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>
"""

import os
import platform

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial)


class ValidMixin(object):
    @staticmethod
    def valid_exp_name(exp_name):
        if exp_name not in ('dissertation', 'stocksp_cor15'):
            raise ValueError('unknown exp_name:{}'.format(exp_name))

    @staticmethod
    def valid_setting(setting):
        if setting not in ("compact", "general"):
            raise ValueError("Unknown setting: {}".format(setting))

    @staticmethod
    def valid_range_value(name, value, lower_bound, upper_bound):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : float or integer
        upper bound : float or integer
        lower_bound : float or integer
        """
        if not lower_bound <= value <= upper_bound:
            raise ValueError("The {}' value {} not in the given bound ({}, "
                             "{}).".format(name, value, upper_bound,
                                           lower_bound))

    @staticmethod
    def valid_nonnegative_value(name, value):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : integer or float
        """
        if value < 0:
            raise ValueError("The {}'s value {} should be nonnegative.".format(
                name, value))

    @staticmethod
    def valid_positive_value(name, value):
        """
        Parameter:
        -------------
        name: string
            name of the value

        value : integer or float
        """
        if value <= 0:
            raise ValueError("The {}'s value {} should be positive.".format(
                name, value))

    @staticmethod
    def valid_nonnegative_list(name, values):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : list[int] or list[float]
        """
        arr = np.asarray(values)
        if np.any(arr < 0):
            raise ValueError("The {} contain negative values.".format(
                name, arr))

    @staticmethod
    def valid_dimension(dim1_name, dim1, dim2):
        """
        Parameters:
        -------------
        dim1, dim2: positive integer
        dim1_name, str
        """
        if dim1 != dim2:
            raise ValueError("mismatch {} dimension: {}, {}".format(
                dim1_name, dim1, dim2))

    @staticmethod
    def valid_trans_date(start_date, end_date):
        """
        Parameters:
        --------------
        start_date, end_date: datetime.date
        """
        if start_date >= end_date:
            raise ValueError("wrong transaction interval, start:{}, "
                             "end:{})".format(start_date, end_date))


class SPSP_Base(ValidMixin):

    def __init__(self,
                 str setting,
                 str group_name,
                 candidate_symbols,
                 int max_portfolio_size,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 int rolling_window_size=200,
                 int n_scenario=200,
                 int scenario_set_idx=1,
                 int print_interval=10):
        """
        stage-wise portfolio stochastic programming basic model

        Parameters:
        -------------
        setting : string,
            {"compact", "general"}

        group_name: string,
            Name of the portfolio

        candidate_symbols : [str],
            The size of the candidate set is n_stock.

        max_portfolio_size : positive integer
            The max number of stock we can invest in the portfolio.
            The model is the mixed integer linear programming, however,
            if the max_portfolio_size == n_stock, it degenerates to the
            linear programming.

        risk_rois : xarray.DataArray,
            dim:(trans_date, symbol),
            shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        risk_free_rois : xarray.DataArray,
            dim: (trans_date),
            shape: (n_exp_period, )
            The return of risk-free asset, usually all zeros.

        initial_risk_wealth : xarray.DataArray, shape: (n_stock,)
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

        rolling_window_size : positive integer
            The historical trading days for estimating statistics.

        n_scenario : positive integer
            The number of scenarios to generate.

        scenario_set_idx :  positive integer
            The index number of scenario set.

        print_interval : positive integer

        Data
        --------------
        decision xarray.DataArray, shape: (n_exp_period, n_stock+1, 5)
        """

        # verify candidate_symbols
        self.valid_dimension("n_symbol", len(candidate_symbols),
                             risk_rois.shape[1])

        self.n_symbol = len(candidate_symbols)
        self.candidate_symbols = candidate_symbols
        self.risk_free_symbol = 'risk_free'
        self.pf_symbols = candidate_symbols + [self.risk_free_symbol, ]

        # pandas.core.indexes.datetimes.DatetimeIndex
        self.all_trans_dates = risk_rois.get_index('trans_date')
        self.n_all_period = len(self.all_trans_dates)

        # verify setting
        if setting not in ("compact", "general"):
            raise (ValueError("Incorrect setting: {}".format(setting)))

        if (setting in ("compact",) and
                max_portfolio_size != self.n_symbol):
            raise (ValueError(
                "The max portfolio size {} must be the same "
                "as the number of symbols {} in compact setting".format(
                    max_portfolio_size, self.n_symbol)))
        self.setting = setting

        # verify group name
        if group_name not in pp.GROUP_SYMBOLS.keys():
            raise ValueError('unknown group name:{}'.format(group_name))
        self.group_name = group_name

        # verify max_portfolio_size
        self.valid_nonnegative_value("max_portfolio_size", max_portfolio_size)
        self.max_portfolio_size = max_portfolio_size

        if max_portfolio_size > self.n_symbol:
            raise (ValueError(
                "The portfolio size {} cannot large than the "
                "size of candidate set. {}.".format(
                    max_portfolio_size, self.n_symbol)))

        # verify risky rois and risk_free_rois
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # verify initial_wealth
        self.valid_dimension("n_symbol", len(candidate_symbols),
                             len(initial_risk_wealth))

        self.valid_nonnegative_list(
            "initial_risk_wealth", initial_risk_free_wealth)
        self.initial_risk_wealth = initial_risk_wealth

        self.valid_nonnegative_value("initial_risk_free_wealth",
                                     initial_risk_free_wealth)
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # verify transaction fee
        self.valid_range_value("buy_trans_fee", buy_trans_fee, 0, 1)
        self.buy_trans_fee = buy_trans_fee

        self.valid_range_value("sell_trans_fee", sell_trans_fee, 0, 1)
        self.sell_trans_fee = sell_trans_fee

        # note that .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)

        # truncate rois to experiment interval
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[
                                  start_date:end_date]

        self.exp_trans_dates = self.exp_risk_rois.get_index('trans_date')
        self.n_exp_period = len(self.exp_trans_dates)
        self.exp_start_date = self.exp_trans_dates[0]
        self.exp_end_date = self.exp_trans_dates[self.n_exp_period - 1]

        self.exp_start_date_idx = self.all_trans_dates.get_loc(
            self.exp_start_date)
        self.exp_end_date_idx = self.all_trans_dates.get_loc(
            self.exp_end_date)

        # # verify rolling_window_size
        self.valid_nonnegative_value("rolling_window_size",
                                     rolling_window_size)
        self.rolling_window_size = int(rolling_window_size)

        # verify n_scenario
        self.valid_nonnegative_value("n_scenario", n_scenario)
        self.n_scenario = int(n_scenario)

        self.valid_nonnegative_value("print_interval", print_interval)
        self.print_interval = print_interval

        # load scenario panel, shape:(n_exp_period, n_stock, n_scenario)
        self.scenario_set_idx = scenario_set_idx
        self.scenario_xarr = self.load_generated_scenario()
        print("scenario shape:", self.scenario_xarr.shape)
        print(self.scenario_xarr)

        # results data
        # decision xarray, shape: (n_exp_period, n_symbol+1, 4)
        decisions = ["wealth", "buy", "sell", "chosen"]
        self.decision_xarr = xr.DataArray(
            np.zeros((self.n_exp_period, self.n_symbol + 1, len(decisions))),
            dims=('trans_date', 'symbol', 'decision'),
            coords=(
                self.exp_trans_dates,
                self.pf_symbols,
                decisions
            )
        )

    def load_generated_scenario(self):
        """
        load generated scenario xarray

        Returns
        ---------------
        scenario_xarr: xarray.DataArray ,
            dims=(trans_date, symbol, sceenario),
            shape: (n_exp_period, n_stock,  n_scenario)
        """

        # portfolio
        scenario_file = pp.SCENARIO_NAME_FORMAT.format(
            group_name=self.group_name,
            n_symbol=self.n_symbol,
            rolling_window_size=self.rolling_window_size,
            n_scenario=self.n_scenario,
            sdx=self.scenario_set_idx,
            scenario_start_date=pp.SCENARIO_START_DATE.strftime("%Y%m%d"),
            scenario_end_date=pp.SCENARIO_END_DATE.strftime("%Y%m%d"),
        )

        scenario_path = os.path.join(pp.SCENARIO_SET_DIR, scenario_file)

        if not os.path.exists(scenario_path):
            raise ValueError("{} not exists.".format(scenario_path))

        # the experiment interval maybe subset of scenarios.
        scenario_xarr = xr.open_dataarray(scenario_path)
        if (self.exp_start_date != pp.SCENARIO_START_DATE or
                self.exp_end_date != pp.SCENARIO_END_DATE):
            # truncate xarr
            scenario_xarr = scenario_xarr.loc[
                            self.exp_start_date:self.exp_end_date]

        if self.n_symbol == 1:
            # the shape of original file is (n_trans_date, n_scenario)
            # reshape to (n_trans_date, symbol(1), n_scenario)
            scenario_xarr = scenario_xarr.expand_dims('symbol', axis=1)

        return scenario_xarr

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        estimating next period risky assets rois,

        Returns:
        ----------------------------
        xarray.DataArray, shape: (n_stock, n_scenario)
        """
        xarr = self.scenario_xarr.loc[kwargs['trans_date']]
        return xarr

    def get_estimated_risk_free_roi(self, *arg, **kwargs):
        """
        estimating next period risk free asset rois,

        Returns:
        ------------------------------
        risk_free_roi : float
        """
        return 0.0

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        the buy amounts and sell amounts of current trans_date are determined
        by the historical data.:
        """
        raise NotImplementedError('get_current_buy_sell_amounts() '
                                  'does not be implemented.')

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
            simulation name of this experiment
        """
        raise NotImplementedError('get_simulation_name() '
                                  'does not be implemented.')

    @staticmethod
    def get_performance_report(
            simulation_name,
            group_name,
            candidate_symbols,
            risk_free_symbol,
            setting,
            max_portfolio_size,
            exp_start_date,
            exp_end_date,
            n_exp_period,
            buy_trans_fee,
            sell_trans_fee,
            initial_wealth,
            final_wealth,
            cum_trans_fee_loss,
            rolling_window_size,
            n_scenario,
            alpha,
            decision_xarr,
            estimated_risk_xarr
    ):
        """
        simulation reports

        Parameters:
        ------------------
        simulation_name : string
        group_name : string
        candidate_symbols: list of string
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
        estimated risk_xarr, xarray.DataArray, shape(n_exp_period, 6)
        """
        reports = dict()

        # basic information
        reports['os_uname'] = "|".join(platform.uname())
        reports['simulation_name'] = simulation_name
        reports['group_name'] = group_name
        reports['candidate_symbols'] = candidate_symbols
        reports['risk_free_symbol'] = risk_free_symbol
        reports['exp_start_date'] = exp_start_date
        reports['exp_end_date'] = exp_end_date
        reports['n_exp_period'] = n_exp_period
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['initial_wealth'] = initial_wealth
        reports['final_wealth'] = final_wealth
        reports['cum_trans_fee_loss'] = cum_trans_fee_loss
        reports['rolling_window_size'] = rolling_window_size
        reports['decision_xarr'] = decision_xarr
        reports['estimated_risk_xarr'] = estimated_risk_xarr

        # analysis
        reports['n_symbol'] = len(candidate_symbols)
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
        run the simulation

        Returns:
        ----------------
        standard report
        """
        raise NotImplementedError('run() does not be implemented.')
