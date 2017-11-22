# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import platform
import numpy as np

from arch.bootstrap.multiple_comparrison import (SPA, )
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial, maximum_drawdown)


class PortfolioReportMixin(object):
    @staticmethod
    def get_performance_report(
            simulation_name, symbols, start_date,
            end_date, buy_trans_fee, sell_trans_fee,
            initial_wealth, final_wealth, n_exp_period,
            trans_fee_loss, wealth_df):
        """
        standard reports

        Parameters:
        ------------------
        simulation_name : string
        symbols: list of string
            the candidate symbols in the simulation
        start_date, end_date: datetime.date
            the starting and ending days of the simulation
        buy_trans_fee, sell_trans_fee: float
            the transaction fee in the simulation
        initial_wealth, final_wealth: float
        n_exp_period: integer
        trans_fee_loss: float
        wealth_df: pandas.DataFrame, shape:(n_exp_period, n_stock + 1)
            the wealth series of each symbols in the simulation.
            It includes the risky and risk-free asset.
        """
        reports = dict()

        # basic information
        reports['os_uname'] = "|".join(platform.uname())
        reports['simulation_name'] = simulation_name
        reports['symbols'] = symbols
        reports['start_date'] = start_date
        reports['end_date'] = end_date
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['initial_wealth'] = initial_wealth
        reports['final_wealth'] = final_wealth
        reports['n_exp_period'] = n_exp_period
        reports['trans_fee_loss'] = trans_fee_loss
        reports['wealth_df'] = wealth_df

        # analysis
        reports['n_stock'] = len(symbols)
        reports['cum_roi'] = final_wealth / initial_wealth - 1.
        reports['daily_roi'] = np.power(final_wealth / initial_wealth,
                                        1. / n_exp_period) - 1

        # wealth_arr, shape: (n_stock+1,)
        wealth_arr = wealth_df.sum(axis=1)
        wealth_daily_rois = wealth_arr.pct_change()
        wealth_daily_rois[0] = 0

        reports['daily_mean_roi'] = wealth_daily_rois.mean()
        reports['daily_std_roi'] = wealth_daily_rois.std()
        reports['daily_skew_roi'] = wealth_daily_rois.skew()

        # excess Kurtosis
        reports['daily_exkurt_roi'] = wealth_daily_rois.kurt()
        reports['sharpe'] = Sharpe(wealth_daily_rois)
        reports['sortino_full'], reports['sortino_full_semi_std'] = \
            Sortino_full(wealth_daily_rois)

        reports['sortino_partial'], reports['sortino_partial_semi_std'] = \
            Sortino_partial(wealth_daily_rois)

        reports['max_abs_drawdown'] = maximum_drawdown(wealth_arr)

        # statistics test
        # SPA test, benchmark is no action
        spa_na = SPA(wealth_daily_rois, np.zeros(wealth_arr.size), reps=1000)
        spa_na.seed(np.random.randint(0, 2 ** 31 - 1))
        spa_na.compute()
        reports['noaction_SPA_l_pvalue'] = spa_na.pvalues[0]
        reports['noaction_SPA_c_pvalue'] = spa_na.pvalues[1]
        reports['noaction_SPA_u_pvalue'] = spa_na.pvalues[2]

        # SPA test, benchmark is buy-and-hold
        spa_bah = SPA(wealth_daily_rois, np.ones(wealth_arr.size), reps=1000)
        spa_bah.seed(np.random.randint(0, 2 ** 31 - 1))
        spa_bah.compute()
        reports['bah_SPA_l_pvalue'] = spa_bah.pvalues[0]
        reports['bah_SPA_c_pvalue'] = spa_bah.pvalues[1]
        reports['bah_SPA_u_pvalue'] = spa_bah.pvalues[2]

        return reports


class ValidMixin(object):
    @staticmethod
    def valid_range_value(name, value, upper_bound, lower_bound):
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
    def valid_positive_value(name, value):
        """
        Parameter:
        -------------
        name: string
            name of the value
        value : integer or float
        """
        if value < 0:
            raise ValueError("The {}'s value {} should be positive.".format(
                name, value))

    @staticmethod
    def valid_positive_list(name, values):
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
