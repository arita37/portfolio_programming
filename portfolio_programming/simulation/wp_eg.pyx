# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import sys

import numpy as np
import xarray as xr
import portfolio_programming as pp
from portfolio_programming.simulation.wp_base import WeightPortfolio


class EGPortfolio(WeightPortfolio):
    """
    exponential gradient strategy

    D. P. Helmbold, R. E. Schapire, Y. Singer, and M. K. Warmuth, "On‐Line
    Portfolio Selection Using Multiplicative Updates," Mathematical Finance,
    vol. 8, pp. 325-347, 1998.
    """

    def __init__(self,
                 double eta,
                 str group_name,
                 list symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth=1e6,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 int print_interval=10):
        super(EGPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval)
        # learning rate
        self.eta = eta

    def get_simulation_name(self, *args, **kwargs):
        return "EG_{}_{}_{}_{}".format(
            self.eta,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

    def add_to_reports(self, reports):
        reports['eta'] = self.eta
        return reports

    def get_today_weights(self, *args, **kwargs):
        """
        remaining the same weight as the today_prev_weights

        Parameters: kwargs
        -------------------------
        prev_trans_date=yesterday,
        trans_date=today,
        today_prev_wealth=today_prev_wealth,
        today_prev_portfolio_wealth=today_prev_portfolio_wealth
        """
        yesterday = kwargs['prev_trans_date']
        today = kwargs['trans_date']

        prev_weights = self.decision_xarr.loc[
            yesterday, self.symbols, 'weight']
        price_relatives = self.exp_rois.loc[today, self.symbols] + 1
        today_prev_weights_sum =  (prev_weights *  price_relatives).sum()
        new_weights = (prev_weights * np.exp(self.eta * price_relatives /
                                             today_prev_weights_sum))
        normalized_new_weights = new_weights / new_weights.sum()
        return normalized_new_weights


class EGAdaptivePortfolio(WeightPortfolio):
    """
    exponential gradient strategy

    D. P. Helmbold, R. E. Schapire, Y. Singer, and M. K. Warmuth, "On‐Line
    Portfolio Selection Using Multiplicative Updates," Mathematical Finance,
    vol. 8, pp. 325-347, 1998.
    """

    def __init__(self,
                 str group_name,
                 list symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth=1e6,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 beta = None,
                 int print_interval=10
                ):
        super(EGAdaptivePortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval)
        # learning rates
        self.etas = xr.DataArray(
            np.zeros(self.n_exp_period),
            dims=('trans_date',),
            coords=(self.exp_trans_dates,)
        )

        self.beta = beta
        self.log_m = np.log(self.n_symbol)


    def get_simulation_name(self, *args, **kwargs):
        if not self.beta:
            return "EG_Adaptive_{}_{}_{}".format(
                self.group_name,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d")
            )
        else:
             return "EG_Adaptive_{}_{}_{}_{}".format(
                self.beta,
                self.group_name,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d")
            )

    def add_to_reports(self, reports):
        if not self.beta:
            reports['beta'] = self.beta
        reports['adaptive_eta'] = self.etas
        return reports

    def get_today_weights(self, *args, **kwargs):
        """
        remaining the same weight as the today_prev_weights

        Parameters: kwargs
        -------------------------
        prev_trans_date=yesterday,
        trans_date=today,
        today_prev_wealth=today_prev_wealth,
        today_prev_portfolio_wealth=today_prev_portfolio_wealth
        """
        yesterday = kwargs['prev_trans_date']
        today = kwargs['trans_date']
        tdx = kwargs['tdx']

        prev_weights = self.decision_xarr.loc[
            yesterday, self.symbols, 'weight']
        # lower bound of historical price relative
        if not self.beta:
            beta = np.min(self.exp_rois.loc[:today, self.symbols]) + 1
        else:
            beta = self.beta
        self.etas.loc[today] = 2*beta*np.sqrt(2* self.log_m/tdx)

        price_relatives = self.exp_rois.loc[today, self.symbols] + 1
        today_prev_weights_sum =  (prev_weights *  price_relatives).sum()
        new_weights = (prev_weights * np.exp( self.etas.loc[today] *
                                             price_relatives /
                                             today_prev_weights_sum))
        normalized_new_weights = new_weights / new_weights.sum()
        return normalized_new_weights


class ExpPortfolio(WeightPortfolio):
    """
    exponential forecaster
    """

    def __init__(self,
                 double eta,
                 str group_name,
                 list symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth=1e6,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 int print_interval=10):
        super(ExpPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval)
        # learning rate
        self.eta = eta

    def get_simulation_name(self, *args, **kwargs):
        return "Exp_{:.2f}_{}_{}_{}".format(
            self.eta,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

    def add_to_reports(self, reports):
        reports['eta'] = self.eta
        return reports

    def get_today_weights(self, *args, **kwargs):
        """
        remaining the same weight as the today_prev_weights

        Parameters: kwargs
        -------------------------
        prev_trans_date=yesterday,
        trans_date=today,
        today_prev_wealth=today_prev_wealth,
        today_prev_portfolio_wealth=today_prev_portfolio_wealth
        """
        yesterday = kwargs['prev_trans_date']
        today = kwargs['trans_date']
        # shape:  n_symbol
        stock_payoffs = (self.exp_rois.loc[:today, self.symbols] + 1).sum(
            axis=0)
        new_weights = np.exp(self.eta * stock_payoffs)
        normalized_new_weights = new_weights / new_weights.sum()
     
        return normalized_new_weights