# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""
import numpy as np
import xarray as xr
import portfolio_programming as pp
from portfolio_programming.simulation.wp_base import (WeightPortfolio,
                                                      NIRUtility)


class PolynomialPortfolio(WeightPortfolio):
    """
    polynomial aggregator strategy
    """

    def __init__(self,
                 poly_power,
                 group_name,
                 symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth=1e6,
                 buy_trans_fee=pp.BUY_TRANS_FEE,
                 sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):

        super(PolynomialPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)
        # power degree
        if poly_power < 1:
            raise ValueError('poly power must >= 1, but get {}'.format(poly_power))

        self.poly_power = poly_power

    def get_simulation_name(self, *args, **kwargs):
        return "Poly_{:.2f}_{}_{}_{}".format(
            self.poly_power,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

    def add_to_reports(self, reports):
        reports['poly_power'] = self.poly_power
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
        today_prev_wealth = kwargs['today_prev_wealth']

        # shape: tdx
        portfolio_payoffs = np.log(self.decision_xarr.loc[:today, self.symbols,
                                 'portfolio_payoff'].sum(axis=1))
        # shape: tdx * n_symbol, does not need take log operation
        # because log(price relative) = simple roi
        stock_payoffs = self.exp_rois.loc[:today, self.symbols]
        # shape: n_symbol
        diff = (stock_payoffs - portfolio_payoffs).sum(axis=0)
        new_weights = np.power(np.maximum(diff, np.zeros_like(diff)),
                               self.poly_power - 1)
        normalized_new_weights = new_weights / new_weights.sum()

        return normalized_new_weights


class NIRPolynomialPortfolio(WeightPortfolio, NIRUtility):
    """
    no internal regret
    """

    def __init__(self,
                 poly_power,
                 group_name,
                 symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth=1e6,
                 buy_trans_fee=pp.BUY_TRANS_FEE,
                 sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 print_interval=10,
                 report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        super(NIRPolynomialPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval, report_dir)
        # power degree
        if poly_power < 1:
            raise ValueError(
                'poly power must >= 1, but get {}'.format(poly_power))

        self.poly_power = poly_power

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
        return "NIRPoly_{:.2f}_{}_{}_{}".format(
            self.poly_power,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

    def add_to_reports(self, reports):
        reports['poly_power'] = self.poly_power
        reports[
            'virtual_expert_decision_xarr'] = self.virtual_expert_decision_xarr
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
        today_prev_wealth = kwargs['today_prev_wealth']
        today_price_relative = kwargs['today_price_relative']

        # record virtual experts' payoff,
        # shape: n_virtual_expert, n_symbol
        self.virtual_expert_decision_xarr.loc[
            today, self.virtual_experts, self.symbols, 'portfolio_payoff'] = (
                self.virtual_expert_decision_xarr.loc[
                    yesterday, self.virtual_experts, self.symbols, 'weight'] *
                today_price_relative
        )

        # shape: tdx
        portfolio_payoffs = np.log(self.decision_xarr.loc[:today, self.symbols,
                                   'portfolio_payoff'].sum(axis=1))
        # shape:  tdx * n_virtual_expert
        virtual_payoffs = np.log(
            self.virtual_expert_decision_xarr.loc[
            :today, self.virtual_experts,
            self.symbols, 'portfolio_payoff'].sum(axis=2)
        )
        # shape: n_virtual_expert
        diff = (virtual_payoffs - portfolio_payoffs).sum(axis=0)
        new_weights = np.power(np.maximum(diff, np.zeros_like(diff)),
                               self.poly_power - 1)
        virtual_expert_weights = new_weights / new_weights.sum()

        # build column stochastic matrix to get weights of today
        S = self.column_stochastic_matrix(self.n_symbol,
                                          virtual_expert_weights.values)
        eigs, eigvs = np.linalg.eig(S)
        # the largest eigvenvalue is 1
        one_index = eigs.argmax()
        normalized_new_weights = (eigvs[:, one_index] /
                                  eigvs[:, one_index].sum()).astype(np.float64)
        # record modified strategies of today
        self.virtual_expert_decision_xarr.loc[
            today, self.virtual_experts, self.symbols, 'weight'
        ] = self.modified_probabilities(normalized_new_weights)

        return normalized_new_weights
