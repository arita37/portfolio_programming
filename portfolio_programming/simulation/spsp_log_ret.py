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

from portfolio_programming.simulation.spsp_base import SPSP_Base


def spsp_log_return(candidate_symbols,
                    setting,
                    max_portfolio_size,
                    risk_rois,
                    risk_free_roi: float,
                    allocated_risk_wealth,
                    allocated_risk_free_wealth,
                    buy_trans_fee,
                    sell_trans_fee,
                    predict_risk_rois,
                    predict_risk_free_roi,
                    n_scenario,
                    solver="ipopt"):
    """
    2nd-stage  stochastic programming.
    The maximize_portfolio_size is equal to the n_stock.
    It will be called in get_current_buy_sell_amounts function.

    Parameters:
    --------------------------
    candidate_symbols: list of string
    setting: string
        {"compact", "general"}
    max_portfolio_size, int
    risk_rois: numpy.array, shape: (n_stock, )
    risk_free_roi: float,
    allocated_risk_wealth: numpy.array, shape: (n_stock,)
    allocated_risk_free_wealth: float
    buy_trans_fee: float
    sell_trans_fee: float
    alpha: float, 1-alpha is the significant level
    predict_risk_ret: numpy.array, shape: (n_stock, n_scenario)
    predict_risk_free_roi: float
    n_scenario: integer
    solver: str, supported by Pyomo

    Returns
    -------------------
    results: dict
        "amounts": xarray.DataArray, shape:(n_symbol, 3),
            coords: (symbol, ('buy', 'sell','chosen'))
    """
    t0 = time()

    n_symbol = len(candidate_symbols)

    # Model
    instance = ConcreteModel()
    instance.max_portfolio_size = max_portfolio_size
    instance.risk_rois = risk_rois
    instance.risk_free_roi = risk_free_roi
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.predict_risk_rois = predict_risk_rois
    # shape: (n_stock,)
    instance.mean_predict_risk_rois = predict_risk_rois.mean(axis=1)
    instance.predict_risk_free_roi = predict_risk_free_roi

    # Set
    instance.symbols = np.arange(n_symbol)
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    # first stage
    instance.buy_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.sell_amounts = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_wealth = Var(instance.symbols, within=NonNegativeReals)
    instance.risk_free_wealth = Var(within=NonNegativeReals)

    # common setting constraint
    def risk_wealth_constraint_rule(model, mdx):
        """
        risk_wealth is a decision variable which depends on both buy_amount
        and sell_amount.
        i.e. the risk_wealth depends on scenario.

        buy_amount and sell_amount are first stage variable,
        risk_wealth is second stage variable.
        """
        return (model.risk_wealth[mdx] == (1. + model.risk_rois[mdx]) *
                model.allocated_risk_wealth[mdx] +
                model.buy_amounts[mdx] - model.sell_amounts[mdx])

    instance.risk_wealth_constraint = Constraint(
        instance.symbols, rule=risk_wealth_constraint_rule)

    # common setting constraint
    def risk_free_wealth_constraint_rule(model):
        total_sell = sum((1. - model.sell_trans_fee) * model.sell_amounts[mdx]
                         for mdx in model.symbols)
        total_buy = sum((1. + model.buy_trans_fee) * model.buy_amounts[mdx]
                        for mdx in model.symbols)

        return (model.risk_free_wealth ==
                (1. + risk_free_roi) * allocated_risk_free_wealth +
                total_sell - total_buy)

    instance.risk_free_wealth_constraint = Constraint(
        rule=risk_free_wealth_constraint_rule)

    # additional variables and setting in the general setting
    if setting == "general":
        # aux variable, switching stock variable
        instance.chosen = Var(instance.symbols, within=Binary)

        # general setting constraint
        def chosen_constraint_rule(model, mdx):
            portfolio_wealth = (sum(model.risk_wealth[idx] for idx in
                                   model.symbols) + model.risk_free_wealth)

            return (model.risk_wealth[mdx] <= model.chosen[mdx] *
                                                portfolio_wealth)

        instance.chosen_constraint = Constraint(instance.symbols,
                                                rule=chosen_constraint_rule)

        # general setting constraint
        def portfolio_size_constraint_rule(model):
            return (sum(model.chosen[mdx] for mdx in model.symbols) <=
                    model.max_portfolio_size)

        instance.portfolio_size_constraint = Constraint(
            rule=portfolio_size_constraint_rule)

    # common setting objective
    def log_return_objective_rule(model):
        portfolio_wealth = log(sum(model.risk_wealth[idx] for idx in
                                model.symbols) + model.risk_free_wealth)

        return portfolio_wealth

    instance.log_ret_objective = Objective(rule=log_return_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # buy and sell amounts
    actions = ['buy', 'sell', 'chosen']
    amounts = xr.DataArray(
        [(instance.buy_amounts[mdx].value,
          instance.sell_amounts[mdx].value,
          -1)
         for mdx in range(n_symbol)],
        dims=('symbol', "action"),
        coords=(candidate_symbols, actions),
    )

    if setting == "general":
        chosens = [instance.chosen[mdx].value for mdx in range(n_symbol)]
    elif setting in ("compact", "compact_mu0"):
        chosens = [1 for _ in range(n_symbol)]

    amounts.loc[candidate_symbols, 'chosen'] = chosens

    return {
        "amounts": amounts,
    }


class SPSP_LogRet(SPSP_Base):
    def __init__(self,
                 setting,
                 group_name,
                 candidate_symbols,
                 max_portfolio_size,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 initial_risk_free_wealth,
                 buy_trans_fee=pp.BUY_TRANS_FEE,
                 sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 rolling_window_size=200,
                 n_scenario=1000,
                 scenario_set_idx=1,
                 print_interval=10):
        """
        stage-wise portfolio stochastic programming max log return model

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
        super(SPSP_LogRet, self).__init__(
            setting,
            group_name,
            candidate_symbols,
            max_portfolio_size,
            risk_rois,
            risk_free_rois,
            initial_risk_wealth,
            initial_risk_free_wealth,
            buy_trans_fee,
            sell_trans_fee,
            start_date,
            end_date,
            rolling_window_size,
            n_scenario,
            scenario_set_idx,
            print_interval
        )