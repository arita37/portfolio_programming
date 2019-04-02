# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import numpy as np
import xarray as xr

import portfolio_programming as pp
from portfolio_programming.simulation.wp_base import (WeightPortfolio,
                                                      NIRUtility)


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
                 int print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        super(EGPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)
        # learning rate
        self.eta = eta

    def get_simulation_name(self, *args, **kwargs):
        return "EG_{:.2f}_{}_{}_{}".format(
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
        today_price_relative = kwargs['today_price_relative']

        prev_weights = self.decision_xarr.loc[
            yesterday, self.symbols, 'weight']
        today_prev_weights_sum = (prev_weights * today_price_relative).sum()
        new_weights = (prev_weights * np.exp(self.eta * today_price_relative /
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
                 int print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR
                 ):
        super(EGAdaptivePortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)
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
            return "EG_Adaptive_{:.2f}_{}_{}_{}".format(
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
            low = np.min(self.exp_rois.loc[:today, self.symbols]) + 1
            high = np.max(self.exp_rois.loc[:today, self.symbols]) + 1
            beta = low / high
        else:
            beta = self.beta
        self.etas.loc[today] = beta * np.sqrt(8 * self.log_m / tdx)

        price_relatives = self.exp_rois.loc[today, self.symbols] + 1
        today_prev_weights_sum = (prev_weights * price_relatives).sum()
        new_weights = (prev_weights * np.exp(self.etas.loc[today] *
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
                 int print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        super(ExpPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)
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
        # does not need take log operation because
        # log(price relative) = simple roi
        stock_payoffs = (self.exp_rois.loc[:today, self.symbols]).sum(axis=0)
        new_weights = np.exp(self.eta * stock_payoffs)
        normalized_new_weights = new_weights / new_weights.sum()

        return normalized_new_weights


class ExpAdaptivePortfolio(WeightPortfolio):
    """
    exponential forecaster
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
                 int print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        super(ExpAdaptivePortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)

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
            return "Exp_Adaptive_{}_{}_{}".format(
                self.group_name,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d")
            )
        else:
            return "Exp_Adaptive_{:.2f}_{}_{}_{}".format(
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

        if not self.beta:
            beta = np.max(self.exp_rois.loc[:today, self.symbols])
        else:
            beta = self.beta
        self.etas.loc[today] = 1 / beta * np.sqrt(8 * self.log_m / tdx)

        # shape:  n_symbol
        # does not need take log operation because
        # log(price relative) = simple roi
        stock_payoffs = (self.exp_rois.loc[:today, self.symbols]).sum(axis=0)
        new_weights = np.exp(self.etas.loc[today] * stock_payoffs)
        normalized_new_weights = new_weights / new_weights.sum()

        return normalized_new_weights


class NIRExpPortfolio(WeightPortfolio, NIRUtility):
    """
    no internal regret exponential forecaster
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
                 int print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        super(NIRExpPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)
        # learning rate
        self.eta = eta

        # fictitious experts,
        self.virtual_experts = ["{}-{}".format(s1, s2)
                                for s1 in self.symbols
                                for s2 in self.symbols
                                if s1 != s2]
        # shape: n_exp_period * (n_symbol * (n_symbol - 1)) * n_symbol *
        # decisions
        decisions = ["weight", 'portfolio_payoff']
        self.virtual_expert_decision_xarr = xr.DataArray(
            np.zeros((self.n_exp_period,
                      len(self.virtual_experts),
                      self.n_symbol,
                      len(decisions)
                      )),
            dims=('trans_date', 'virtual_experts', 'symbol', 'decision'),
            coords=(
                self.exp_trans_dates,
                self.virtual_experts,
                self.symbols,
                decisions
            )
        )

    def get_simulation_name(self, *args, **kwargs):
        return "NIRExp_{:.2f}_{}_{}_{}".format(
            self.eta,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

    def add_to_reports(self, reports):
        reports['eta'] = self.eta
        reports['virtual_expert_decision_xarr'] = self.virtual_expert_decision_xarr
        return reports

    def pre_trading_operation(self, *args, **kargs):
        """
        operations after initialization and before trading
        """
        today = self.exp_start_date
        self.virtual_expert_decision_xarr.loc[
            today,
            self.virtual_experts,
            self.symbols,
            'weight'
        ] = self.modified_probabilities(self.initial_weights)

        # the portfolio payoff of first decision
        self.virtual_expert_decision_xarr.loc[
            today,
            self.virtual_experts,
            self.symbols,
            'portfolio_payoff'
        ] = self.modified_probabilities(self.initial_weights)


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
        today_price_relative = kwargs['today_price_relative']

        # record virtual experts' payoff,
        # shape: n_virtual_expert, n_symbol
        self.virtual_expert_decision_xarr.loc[
            today, self.virtual_experts, self.symbols, 'portfolio_payoff'] = (
            self.virtual_expert_decision_xarr.loc[
                yesterday, self.virtual_experts, self.symbols, 'weight']  *
            today_price_relative
        )

        # cumulative returns of all virtual experts
        # first sum: shape: tdx * n_virtual_expert
        # second sum: shape: n_virtual_expert
        virtual_cum_payoffs = np.log(
            self.virtual_expert_decision_xarr.loc[
                :today,self.virtual_experts,
            self.symbols, 'portfolio_payoff'].sum(axis=2)
        ).sum(axis=0)

        # exponential predictors
        new_weights = np.exp(self.eta * virtual_cum_payoffs)

        # normalized weights of virtual experts
        virtual_expert_weights = new_weights / new_weights.sum()

        # build column stochastic matrix to get weights of today
        S = self.column_stochastic_matrix(self.n_symbol,
                                           virtual_expert_weights.values)
        eigs, eigvs = np.linalg.eig(S)
        normalized_new_weights = eigvs[:, 0] / eigvs[:, 0].sum()

        # record modified strategies of today
        self.virtual_expert_decision_xarr.loc[
            today, self.virtual_experts, self.symbols, 'weight'
        ] = self.modified_probabilities(normalized_new_weights)

        return normalized_new_weights





