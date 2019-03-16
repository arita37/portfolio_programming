# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>
"""

import os
import pickle
import platform
from time import time

import numpy as np
import xarray as xr
from pyomo.environ import *

import portfolio_programming as pp
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial)

from portfolio_programming.simulation.spsp_base import ValidMixin


class Experts(object):
    """
    experts' advices based on profit and loss
    """

    def __init__(self, trans_dates, expert_names):
        """
        Parameters:
        -------------
        trans_dates: array_likes

        expert_names: List[str],
            All expert's name

        """
        attrs = ['roi', ]
        self.expert_names = expert_names
        self.n_expert = len(expert_names)
        self.expert_rois = xr.DataArray(
            np.zeros((len(trans_dates), len(expert_names), len(attrs))),
            dims=('trans_date', 'expert', 'roi'),
            coords=(trans_dates, expert_names, attrs)
        )

    def get_expert_rois(self):
        return self.expert_rois

    def set_expert_roi(self, trans_date, rois):
        """
        record the roi of each expert's decision

        rois : xarray.dataarray
        """
        self.expert_rois.loc[trans_date, : , 'roi'] = rois

    def get_advice(self, trans_date):
        pass

    def _uniform(self, trans_date):
        """
        always give the uniform weight
        """
        return xr.DataArray(
            np.ones(self.n_expert)/self.n_expert,
            dims=('expert',),
            coords=(self.expert_names,)
        )

    def _exponential_gradient(self, trans_date, learn_rate=None):
        """
        Parameters:
        -------------
        learn_rate: float or None
            if the learn_rate is None, it will use the adaptive parameter.

        """


    def _no_internal_regret(self, trans_date):
        pass


class SPSP_CVaR_Expert(ValidMixin):
    def __init__(self,
                 group_name,
                 candidate_symbols,
                 experts,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=pp.BUY_TRANS_FEE,
                 sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 n_scenario=1000,
                 scenario_set_idx=1,
                 print_interval=10):
        """
        multi-experts stage-wise portfolio stochastic programming  model

        Parameters:
        -------------
        setting : string,
            {"compact", "general"}

        group_name: string,
            Name of the portfolio

        candidate_symbols : [str],
            The size of the candidate set is n_stock.

        experts: {name:params}
            The parameters of each expert

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

        n_scenario : positive integer
            The number of scenarios to generate.

        scenario_set_idx :  positive integer
            The index number of scenario set.

        print_interval : positive integer


        Data
        --------------
        decision xarray.DataArray, shape: (n_exp_period, n_stock+1, 5)

        """
        # verify group name
        if group_name not in pp.GROUP_SYMBOLS.keys():
            raise ValueError('unknown group name:{}'.format(group_name))
        self.group_name = group_name

        # verify candidate_symbols
        self.valid_dimension("n_symbol", len(candidate_symbols),
                             risk_rois.shape[1])

        self.n_symbol = len(candidate_symbols)
        self.candidate_symbols = candidate_symbols
        self.risk_free_symbol = 'risk_free'
        self.pf_symbols = candidate_symbols + [self.risk_free_symbol, ]

        # experts
        self.experts = experts
        self.n_expert = len(experts)

        # pandas.core.indexes.datetimes.DatetimeIndex
        self.all_trans_dates = risk_rois.get_index('trans_date')
        self.n_all_period = len(self.all_trans_dates)

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

        # estimated risks, shape(n_exp_period, 6)
        risks = ['CVaR', 'VaR', 'EV_CVaR', 'EV_VaR', 'EEV_CVaR', 'VSS']
        self.estimated_risk_xarr = xr.DataArray(
            np.zeros((self.n_exp_period, len(risks))),
            dims=('trans_date', 'risk'),
            coords=(
                self.exp_trans_dates,
                risks
            )
        )



if __name__ == '__main__':
    pass


