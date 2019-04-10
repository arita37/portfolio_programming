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
import pickle
import platform
from time import time
import logging

cimport numpy as cnp
import numpy as np
import xarray as xr
from pyomo.environ import *
import multiprocess as mp

import portfolio_programming as pp
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial)

from portfolio_programming.simulation.spsp_base import (ValidMixin, SPSPBase)
from portfolio_programming.simulation.wp_base import (NIRUtility, )

def spsp_cvar(candidate_symbols,
              str setting,
              int max_portfolio_size,
              cnp.ndarray[cnp.float64_t, ndim=1] risk_rois,
              double risk_free_roi,
              cnp.ndarray[cnp.float64_t, ndim=1] allocated_risk_wealth,
              double allocated_risk_free_wealth,
              double buy_trans_fee,
              double sell_trans_fee,
              double alpha,
              cnp.ndarray[cnp.float64_t, ndim=2] predict_risk_rois,
              double predict_risk_free_roi,
              int n_scenario,
              str solver=pp.PROG_SOLVER):
    """
    2nd-stage minimize CVaR stochastic programming.
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
        "estimated_var": float
        "estimated_cvar": float
        "estimated_ev_var": float
        "estimated_ev_cvar": float
        "estimated_eev_cvar": float
        "vss": vss, float
    """
    t0 = time()

    cdef Py_ssize_t n_symbol = len(candidate_symbols)

    # Model
    instance = ConcreteModel()
    instance.max_portfolio_size = max_portfolio_size
    instance.risk_rois = risk_rois
    instance.risk_free_roi = risk_free_roi
    instance.allocated_risk_wealth = allocated_risk_wealth
    instance.allocated_risk_free_wealth = allocated_risk_free_wealth
    instance.buy_trans_fee = buy_trans_fee
    instance.sell_trans_fee = sell_trans_fee
    instance.alpha = alpha
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

    # aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    instance.Z = Var()

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)

    # common setting constraint
    def risk_wealth_constraint_rule(model, int mdx):
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

    # common setting constraint
    def scenario_constraint_rule(model, int sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        predict_risk_wealth = sum((1. + model.predict_risk_rois[mdx, sdx]) *
                                  model.risk_wealth[mdx]
                                  for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - predict_risk_wealth)

    instance.scenario_constraint = Constraint(instance.scenarios,
                                              rule=scenario_constraint_rule)

    # additional variables and setting in the general setting
    if setting == "general":
        # aux variable, switching stock variable
        instance.chosen = Var(instance.symbols, within=Binary)

        # general setting constraint
        def chosen_constraint_rule(model, int mdx):
            portfolio_wealth = (sum(model.risk_wealth[idx] for idx in
                                    model.symbols) + model.risk_free_wealth)

            # portfolio_wealth = (sum(model.allocated_risk_wealth) +
            #                     model.allocated_risk_free_wealth *
            #                     (1. + model.risk_rois[mdx]))
            return (model.risk_wealth[mdx] <=
                    model.chosen[mdx] * portfolio_wealth)

        instance.chosen_constraint = Constraint(instance.symbols,
                                                rule=chosen_constraint_rule)

        # general setting constraint
        def portfolio_size_constraint_rule(model):
            return (sum(model.chosen[mdx] for mdx in model.symbols) <=
                    model.max_portfolio_size)

        instance.portfolio_size_constraint = Constraint(
            rule=portfolio_size_constraint_rule)

    # common setting objective
    def cvar_objective_rule(model):
        scenario_exp = (sum(model.Ys[sdx] for sdx in range(n_scenario)) /
                        n_scenario)
        return model.Z - 1. / (1. - model.alpha) * scenario_exp

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # logging.DEBUG(display(instance))
    # display(instance)

    # buy and sell amounts
    actions = ['buy', 'sell', 'wealth', 'chosen']
    amounts = xr.DataArray(
        [(instance.buy_amounts[mdx].value,
          instance.sell_amounts[mdx].value,
          instance.risk_wealth[mdx].value,
          -1)
         for mdx in range(n_symbol)],
        dims=('symbol', "action"),
        coords=(candidate_symbols, actions),
    )
    # print(amounts)

    # value at risk (estimated)
    cdef double estimated_var = instance.Z.value
    cdef double estimated_cvar = instance.cvar_objective()

    # expected value (EV) programming
    # delete old CVaR constraint
    instance.del_component("scenario_constraint")
    instance.del_component("cvar_objective")
    instance.del_component("Ys")
    instance.del_component("Ys_index")

    # new scenario variable
    instance.Y = Var(within=NonNegativeReals)

    # update scenario constraint
    def expected_scenario_constraint_rule(model):
        """
        auxiliary variable Y depends on scenario. CVaR <= VaR
        EV only consider expected scenario instead of all scenarios.
        """
        predict_wealth = sum((1. + model.mean_predict_risk_rois[mdx]) *
                             model.risk_wealth[mdx]
                             for mdx in model.symbols)

        return model.Y >= (model.Z - predict_wealth)

    instance.expected_scenario_constraint = Constraint(
        rule=expected_scenario_constraint_rule)

    # update objective
    def ev_cvar_objective_rule(model):
        return model.Z - 1. / (1. - model.alpha) * model.Y

    instance.ev_cvar_objective = Objective(rule=ev_cvar_objective_rule,
                                           sense=maximize)

    # solve eev 1st stage
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # value at risk (estimated)
    cdef double estimated_ev_var = instance.Z.value
    cdef double estimated_ev_cvar = instance.ev_cvar_objective()

    # expected EV (EEV) programming.
    # The EEV used all first stage solution of decision variables of EV, and
    # all scenarios to get the solution.
    # we only need to compute the objective value of EEV.

    # fixed the first-stage variables
    for mdx in instance.symbols:
        instance.buy_amounts[mdx].fixed = True
        instance.sell_amounts[mdx].fixed = True
        instance.risk_wealth[mdx].fixed = True
    instance.risk_free_wealth.fixed = True

    # Z is the first-stage variable but not the second one.
    instance.Z.fixed = True

    # compute EEV objective
    estimated_eev_ys = np.zeros(n_scenario)
    for sdx in range(n_scenario):
        # shape: (n_stock,)
        scen_roi = predict_risk_rois[:, sdx]
        portfolio_wealth = (
                sum((1 + scen_roi[mdx]) * instance.risk_wealth[mdx].value
                    for mdx in np.arange(n_symbol)) +
                instance.risk_free_wealth.value)

        if estimated_var <= portfolio_wealth:
            estimated_eev_ys[sdx] = estimated_var
        else:
            diff = (estimated_var - portfolio_wealth)
            estimated_eev_ys[sdx] = (estimated_var -
                                     1 / (1 - alpha) * diff)

    estimated_eev_cvar = estimated_eev_ys.mean()
    vss = estimated_cvar - estimated_eev_cvar

    chosen_symbols = None
    if setting == "general":
        chosens = [instance.chosen[mdx].value for mdx in range(n_symbol)]
    elif setting in ("compact", "compact_mu0"):
        chosens = [1 for mdx in range(n_symbol)]

    amounts.loc[candidate_symbols, 'chosen'] = chosens

    logging.debug("spsp_cvar {} OK, {:.3f} secs".format(
        setting, time() - t0))

    return {
        "amounts": amounts,
        'risk_free_wealth': instance.risk_free_wealth.value,
        "VaR": estimated_var,
        "CVaR": estimated_cvar,
        "EV_VaR": estimated_ev_var,
        "EV_CVaR": estimated_ev_cvar,
        "EEV_CVaR": estimated_eev_cvar,
        "VSS": vss,
    }


class SPSP_CVaR(SPSPBase):
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
                 double alpha=0.05,
                 int scenario_set_idx=1,
                 int print_interval=10,
                 str report_dir=pp.WEIGHT_PORTFOLIO_REPORT_DIR):
        """
        stage-wise portfolio stochastic programming  model

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

        alpha : float
            The risk-averse level.

        scenario_set_idx :  positive integer
            The index number of scenario set.

        print_interval : positive integer


        Data
        --------------
        decision xarray.DataArray, shape: (n_exp_period, n_stock+1, 5)
        estimated risk_xarr, xarray.DataArray, shape(n_exp_period, 6)

        """
        super(SPSP_CVaR, self).__init__(
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
            print_interval,
            report_dir
        )

        # verify alpha
        self.valid_range_value("alpha", alpha, 0, 1)
        self.alpha = float(alpha)

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

    def get_current_buy_sell_amounts(self, *args, **kwargs):
        """
        the buy amounts and sell amounts of current trans_date are determined
        by the historical data.

        Returns:
        --------------
        results: dict
            "amounts": xarray.DataArray, shape:(n_symbol, 3),
                coords: (symbol, ('buy', 'sell','chosen'))
            "estimated_var": float
            "estimated_cvar": float
            "estimated_ev_var": float
            "estimated_ev_cvar": float
            "estimated_eev_cvar": float
            "vss": vss, float
        """
        # current exp_period index
        trans_date = kwargs['trans_date']
        results = spsp_cvar(
            self.candidate_symbols,
            self.setting,
            self.max_portfolio_size,
            self.exp_risk_rois.loc[trans_date, :].values,
            self.risk_free_rois.loc[trans_date],
            kwargs['allocated_risk_wealth'].values,
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.alpha,
            kwargs['estimated_risk_rois'].values,
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
            pp.PROG_SOLVER,
        )
        return results

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
           simulation name of this experiment
        """

        name = (
            "SPSP_CVaR_{}_{}_Mc{}_M{}_h{}_s{}_a{:.2f}_sdx{}_{}_{}".format(
                self.setting,
                self.group_name,
                self.n_symbol,
                self.max_portfolio_size,
                self.rolling_window_size,
                self.n_scenario,
                self.alpha,
                self.scenario_set_idx,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d"),
            )
        )
        return name

    def run(self):
        """
        run the simulation

        Returns:
        ----------------
        standard report
        """
        t0 = time()

        # get simulation name
        simulation_name = self.get_simulation_name()

        # initial wealth of each stock in the portfolio
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth
        cum_trans_fee_loss = 0

        for tdx in range(self.n_exp_period):
            t1 = time()
            curr_date = self.exp_trans_dates[tdx]

            estimated_risk_rois = self.get_estimated_risk_rois(
                trans_date=curr_date)

            # estimating next period risk_free roi, return float
            estimated_risk_free_roi = self.get_estimated_risk_free_roi()

            # determining the buy and sell amounts
            pg_results = self.get_current_buy_sell_amounts(
                trans_date=curr_date,
                estimated_risk_rois=estimated_risk_rois,
                estimated_risk_free_roi=estimated_risk_free_roi,
                allocated_risk_wealth=allocated_risk_wealth,
                allocated_risk_free_wealth=allocated_risk_free_wealth
            )

            # amount_xarr, dims=('symbol', "amount"),
            amount_xarr = pg_results["amounts"]
            for act in ('buy', 'sell', 'chosen'):
                # the symbol does not contain risk_free symbol
                self.decision_xarr.loc[curr_date, self.candidate_symbols, act] \
                    = amount_xarr.loc[self.candidate_symbols, act]

            # # record the transaction loss
            buy_sum = amount_xarr.loc[:, 'buy'].sum()
            sell_sum = amount_xarr.loc[:, 'sell'].sum()
            cum_trans_fee_loss += (
                    buy_sum * self.buy_trans_fee +
                    sell_sum * self.sell_trans_fee
            )

            # buy and sell amounts consider the transaction cost
            total_buy = (buy_sum * (1 + self.buy_trans_fee))
            total_sell = (sell_sum * (1 - self.sell_trans_fee))

            # capital allocation
            self.decision_xarr.loc[curr_date, self.candidate_symbols, 'wealth'] \
                = (
                    (1 + self.exp_risk_rois.loc[
                        curr_date, self.candidate_symbols]) *
                    allocated_risk_wealth +
                    self.decision_xarr.loc[curr_date,
                                           self.candidate_symbols, 'buy'] -
                    self.decision_xarr.loc[curr_date,
                                           self.candidate_symbols, 'sell']
            )
            self.decision_xarr.loc[
                curr_date, self.risk_free_symbol, 'wealth'] = (
                    (1 + self.exp_risk_free_rois.loc[curr_date]) *
                    allocated_risk_free_wealth -
                    total_buy + total_sell
            )

            # update wealth
            allocated_risk_wealth = self.decision_xarr.loc[
                curr_date, self.candidate_symbols, 'wealth']
            allocated_risk_free_wealth = self.decision_xarr.loc[
                curr_date, self.risk_free_symbol, 'wealth']

            # record risks
            for col in ("VaR", "CVaR", "EV_VaR", "EV_CVaR", "EEV_CVaR", "VSS"):
                self.estimated_risk_xarr.loc[curr_date, col] = pg_results[col]

            # record chosen symbols
            if tdx % self.print_interval == 0:
                logging.info("{} [{}/{}] {} "
                             "wealth:{:.2f}, {:.3f} secs".format(
                    simulation_name,
                    tdx + 1,
                    self.n_exp_period,
                    curr_date.strftime("%Y%m%d"),
                    float(self.decision_xarr.loc[curr_date, :, 'wealth'].sum()),
                    time() - t1)
                )

        # end of simulation, computing statistics
        edx = self.n_exp_period - 1
        initial_wealth = (
                self.initial_risk_wealth.sum() + self.initial_risk_free_wealth)
        final_wealth = self.decision_xarr.loc[self.exp_end_date, :,
                       'wealth'].sum()

        # get reports
        reports = self.get_performance_report(
            simulation_name,
            self.group_name,
            self.candidate_symbols,
            self.risk_free_symbol,
            self.setting,
            self.max_portfolio_size,
            self.exp_start_date,
            self.exp_end_date,
            self.n_exp_period,
            self.buy_trans_fee,
            self.sell_trans_fee,
            float(initial_wealth),
            float(final_wealth),
            float(cum_trans_fee_loss),
            self.rolling_window_size,
            self.n_scenario,
            self.alpha,
            self.decision_xarr,
            self.estimated_risk_xarr
        )

        # add simulation time
        reports['simulation_time'] = time() - t0

        # write report
        report_path = os.path.join(
            self.report_dir,
            "report_{}.pkl".format(self.get_simulation_name())
        )

        with open(report_path, 'wb') as fout:
            pickle.dump(reports, fout, pickle.HIGHEST_PROTOCOL)

        print("{}-{} {} OK, {:.4f} secs".format(
            platform.node(),
            os.getpid(),
            simulation_name,
            time() - t0)
        )

        return reports


class NER_SPSP_CVaR(ValidMixin):
    def __init__(self,
                 str nr_strategy,
                 double nr_strategy_param,
                 str expert_group_name,
                 list experts,
                 str group_name,
                 list candidate_symbols,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 int n_scenario=1000,
                 int scenario_set_idx=1,
                 int print_interval=2,
                 report_dir=pp.NRSPSPCVaR_DIR,
                 ):
        """
        no external regret stage-wise portfolio stochastic programming model

        Parameters:
        -------------
        nr_strategy: string,
            name of the no_regret strategy

        nr_strategy_params: double,
            List of the no-regret strategy's parameters

        expert_group_name: string
            name of the group of the experts

        experts: [(rolling_window_size, alpha)]
            the pair of rolling window_size and alpha.

        group_name: string,
            Name of the portfolio

        candidate_symbols : [str],
            The size of the candidate set is n_stock.

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

        rolling_window_sizes : list of  positive integer
            The historical trading days for estimating statistics.

        n_scenario : positive integer
            The number of scenarios to generate.

        alphas : list of float
            The risk-averse level.

        report_path : string
            The performance report file path of the simulation.

        is_parallel: bool
            Does parallel solve the experts

        Data
        --------------
        decision xarray.DataArray, shape: (n_exp_period, n_stock+1, 5)
        estimated risk_xarr, xarray.DataArray, shape(n_exp_period, 6)

        """
        # no-regret strategy
        if nr_strategy not in ('EG', "POLY", "EXP"):
            raise ValueError('unknown no-regret strategy:', nr_strategy)
        self.nr_strategy = nr_strategy

        self.valid_nonnegative_value('nr_strategy_param', nr_strategy_param)
        self.nr_strategy_param = nr_strategy_param

        # expert_group_name
        self.expert_group_name = expert_group_name
        self.experts = experts
        # experts (rolling_window_size, alpha)
        self.expert_names = ["h{}a{:.2f}".format(h, a)
                             for h, a in experts]
        print(self.expert_names)
        # parameters, rolling_window_size * alpha
        self.n_expert = len(experts)

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

        # verify group name
        if group_name not in pp.GROUP_SYMBOLS.keys():
            raise ValueError('unknown group name:{}'.format(group_name))
        self.group_name = group_name

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
        # exp_risk_rois, shape: (n_exp_period, n_symbol)
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
        distinct_rolling_window_sizes = set(h for h, _ in experts)
        self.scenario_xarr = xr.DataArray(
            np.zeros((len(distinct_rolling_window_sizes),
                      self.n_exp_period,
                      self.n_symbol,
                      self.n_scenario)),
            dims=('rolling_window_size', 'trans_date', 'symbol', 'scenario'),
            coords=(list(distinct_rolling_window_sizes),
                    self.exp_trans_dates,
                    candidate_symbols,
                    np.arange(n_scenario)
                    )
        )

        t0 = time()
        for h in distinct_rolling_window_sizes:
            self.scenario_xarr.loc[h] = self.load_generated_scenario(h)
        print("group:{}, expert:{} scenario shape:{}, {:.3f} secs".format(
            group_name, expert_group_name,
            self.scenario_xarr.shape, time() - t0))

        # results data
        # decision xarray, shape: (n_exp_period, n_expert+1, n_symbol+1, 4)
        # the plus one expert is the final weighted-decision
        # the plus one symbol is the risk-free one
        decisions = ["wealth", "buy", "sell"]
        self.decision_xarr = xr.DataArray(
            np.zeros((self.n_exp_period, self.n_expert + 1,
                      self.n_symbol + 1, len(decisions))),
            dims=('trans_date', 'expert', 'symbol', 'decision'),
            coords=(
                self.exp_trans_dates,
                self.expert_names + ['main', ],
                self.pf_symbols,
                decisions
            )
        )
        # portfolio_properties, shape(n_exp_period, n_expert+1,
        # portfolio_properties)
        portfolio_properties = [
            'wealth', 'tax_loss', 'price_relative', "weight",
            'CVaR', 'VaR', 'EV_CVaR', 'EV_VaR', 'EEV_CVaR', 'VSS']
        self.portfolio_xarr = xr.DataArray(
            np.zeros((self.n_exp_period, self.n_expert + 1,
                      len(portfolio_properties))),
            dims=('trans_date', 'expert', 'property'),
            coords=(
                self.exp_trans_dates,
                self.expert_names + ['main', ],
                portfolio_properties
            )
        )

        # report path
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        self.report_dir = report_dir

    def no_regret_strategy(self, *args, **kwargs):
        """
        determine the weight on each expert.
        """
        tdx = kwargs['tdx']
        yesterday = kwargs['yesterday']
        today = kwargs['today']
        allocated_risk_wealth = kwargs['allocated_risk_wealth']
        allocated_risk_free_wealth = kwargs['allocated_risk_free_wealth']

        prev_main_wealth = (allocated_risk_wealth.sum() +
                            allocated_risk_free_wealth)
        if tdx == 0:
            # price relatives of all experts (considered trans. fee)
            self.portfolio_xarr.loc[today, self.expert_names,
                                    'price_relative'] = (
                    ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                        today, self.candidate_symbols])
                      ).sum() +
                     allocated_risk_free_wealth * (
                             1 + self.risk_free_rois.loc[today]
                     )) / prev_main_wealth
            )
            # price relative of the main portfolio
            self.portfolio_xarr.loc[today, 'main', 'price_relative'] = (
                    ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                        today, self.candidate_symbols])
                      ).sum() +
                     allocated_risk_free_wealth * (
                             1 + self.risk_free_rois.loc[today]
                     )) / prev_main_wealth
            )

            # initial weights
            return 1. / self.n_expert

        else:
            # tdx >= 1

            # price relatives of all experts
            # shape: (n_expert, n_symbol)
            prev_risky_wealths = self.decision_xarr.loc[
                yesterday, self.expert_names,
                self.candidate_symbols, 'wealth']
            # shape: (n_expert,)
            prev_riskfree_wealths = self.decision_xarr.loc[
                yesterday, self.expert_names,
                self.risk_free_symbol, 'wealth']

            curr_risky_wealths = (
                    prev_risky_wealths *
                    (1 + self.exp_risk_rois.loc[
                        today, self.candidate_symbols])
            )
            curr_riskfree_wealths = (
                    prev_riskfree_wealths *
                    (1 + self.risk_free_rois.loc[today])
            )

            # shape: (n_expert,)
            expert_price_relatives = (
                    (curr_risky_wealths.sum(axis=1) + curr_riskfree_wealths) /
                    (prev_risky_wealths.sum(axis=1) + prev_riskfree_wealths)
            )
            # print("curr_risky_wealths:", curr_risky_wealths)
            # print("curr_riskfree_wealths:",  curr_riskfree_wealths)
            # print("expert_price_relatives:", expert_price_relatives)

            self.portfolio_xarr.loc[today, self.expert_names,
                                    'price_relative'] = expert_price_relatives

            # main price relative
            self.portfolio_xarr.loc[today, 'main', 'price_relative'] = (
                    ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                        today, self.candidate_symbols])
                      ).sum() +
                     allocated_risk_free_wealth * (
                             1 + self.risk_free_rois.loc[today]
                     )) / prev_main_wealth
            )

        # tdx >= 1
        if self.nr_strategy == 'EG':
            # shape:  (n_expert,)
            prev_weights = self.portfolio_xarr.loc[
                yesterday, self.expert_names, 'weight']
            price_relatives = self.portfolio_xarr.loc[
                today, self.expert_names, 'price_relative']

            new_weights = prev_weights * np.exp(self.nr_strategy_param *
                                                price_relatives / np.dot(
                prev_weights, price_relatives))

            return new_weights / new_weights.sum()

        elif self.nr_strategy == 'EXP':
            # shape:  (n_expert,)
            expert_payoffs = np.log(
                self.portfolio_xarr.loc[:today,
                self.expert_names, 'price_relative']
            ).sum(axis=0)
            new_weights = np.exp(self.nr_strategy_param * expert_payoffs)
            return new_weights / new_weights.sum()

        elif self.nr_strategy == 'POLY':
            # shape: (tdx, n_expert)
            time_payoffs = np.log(
                self.portfolio_xarr.loc[:today, self.expert_names,
                'price_relative'])
            # shape: (tdx,)
            main_payoffs = np.log(
                self.portfolio_xarr.loc[:today, "main", 'price_relative'])
            # shape: (n_expert,)
            diffs = (time_payoffs - main_payoffs).sum(axis=0)

            # print("time_payoffs:", time_payoffs)
            # print("main_payoffs:", main_payoffs)
            # print(" diffs:",  diffs)
            new_weights = np.power(np.maximum(diffs, np.zeros_like(diffs)),
                                   self.nr_strategy_param - 1)
            return new_weights / new_weights.sum()

    def load_generated_scenario(self, rolling_window_size):
        """
        load generated scenario xarray

        Returns
        ---------------
        scenario_xarr: xarray.DataArray ,
            dims=(trans_date, symbol, sceenario),
            shape: (n_exp_period, n_symbol,  n_scenario)
        """

        # portfolio
        scenario_file = pp.SCENARIO_NAME_FORMAT.format(
            group_name=self.group_name,
            n_symbol=self.n_symbol,
            rolling_window_size=rolling_window_size,
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

        return scenario_xarr

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        estimating next period risky assets rois,

        Returns:
        ----------------------------
        scenarios on the trans_date
        xarray.DataArray, shape: (n_symbol, n_scenario)
        """
        xarr = self.scenario_xarr.loc[kwargs['rolling_window_size'],
                                      kwargs['trans_date']]
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
        by the historical data.

        Returns:
        --------------
        results: dict
            "amounts": xarray.DataArray, shape:(n_symbol, 4),
                coords: (symbol, ('wealth', 'buy', 'sell', 'chosen'))
            "estimated_var": float
            "estimated_cvar": float
            "estimated_ev_var": float
            "estimated_ev_cvar": float
            "estimated_eev_cvar": float
            "vss": vss, float
        """
        # current exp_period index
        trans_date = kwargs['trans_date']
        results = spsp_cvar(
            self.candidate_symbols,
            "compact",
            self.n_symbol,
            self.exp_risk_rois.loc[trans_date, self.candidate_symbols].values,
            self.risk_free_rois.loc[trans_date],
            kwargs['allocated_risk_wealth'].values,
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            kwargs['alpha'],
            kwargs['estimated_risk_rois'].values,
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
            pp.PROG_SOLVER,
        )
        return results

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
           simulation name of this experiment
        """

        name = (
            "NR_SPSP_CVaR_{}_{:.2f}_{}_{}_s{}_sdx{}_{}_{}".format(
                self.nr_strategy,
                self.nr_strategy_param,
                self.group_name,
                self.expert_group_name,
                self.n_scenario,
                self.scenario_set_idx,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d"),
            )
        )
        return name


    @staticmethod
    def get_performance_report(
            simulation_name,
            group_name,
            candidate_symbols,
            risk_free_symbol,
            exp_start_date,
            exp_end_date,
            n_exp_period,
            buy_trans_fee,
            sell_trans_fee,
            initial_wealth,
            final_wealth,
            nr_strategy,
            nr_strategy_param,
            expert_group_name,
            experts,
            n_scenario,
            decision_xarr,
            portfolio_xarr
    ):
        """
        simulation reports
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
        reports['nr_strategy'] = nr_strategy
        reports['nr_strategy_param'] = nr_strategy_param
        reports['expert_group_name'] = expert_group_name
        reports['experts'] = experts

        reports['decision_xarr'] = decision_xarr
        reports['portfolio_xarr'] = portfolio_xarr

        # analysis
        reports['n_symbol'] = len(candidate_symbols)
        reports['cum_roi'] = final_wealth / initial_wealth - 1.
        reports['daily_roi'] = np.power(final_wealth / initial_wealth,
                                        1. / n_exp_period) - 1

        # wealth_arr, Pandas.Series, shape: (n_symbol+1,)
        wealth_arr = portfolio_xarr.loc[:, 'main', 'wealth'].to_series()
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
        t0 = time()

        # get simulation name
        simulation_name = self.get_simulation_name()

        # initial wealth of each stock in the portfolio
        # shape: (n_symbol,), float
        allocated_risk_wealth = self.initial_risk_wealth
        allocated_risk_free_wealth = self.initial_risk_free_wealth

        for tdx in range(self.n_exp_period):
            t1 = time()
            today = self.exp_trans_dates[tdx]
            # print('allocated wealth:',  allocated_risk_wealth)
            # print('allocated_risk_free_wealth:', allocated_risk_free_wealth)

            # experts
            for h, a in self.experts:
                expert_name = "h{}a{:.2f}".format(h, a)

                # fetch scenarios, shape: (n_symbol, n_scenario)
                estimated_risk_rois = self.get_estimated_risk_rois(
                    rolling_window_size=h,
                    trans_date=today)

                # estimating next period risk_free roi, return float
                estimated_risk_free_roi = (
                    self.get_estimated_risk_free_roi()
                )
                # determining the buy and sell amounts
                pg_results = self.get_current_buy_sell_amounts(
                    trans_date=today,
                    alpha=a,
                    estimated_risk_rois=estimated_risk_rois,
                    estimated_risk_free_roi=estimated_risk_free_roi,
                    allocated_risk_wealth=allocated_risk_wealth,
                    allocated_risk_free_wealth=allocated_risk_free_wealth
                )
                # print(expert_name, pg_results)

                # amount_xarr, shape"(n_symbol,
                # ['buy', 'sell', 'wealth', 'chosen']),
                amount_xarr = pg_results["amounts"]
                acts = ['buy', 'sell']
                self.decision_xarr.loc[today, expert_name,
                                       self.candidate_symbols, acts] = (
                    amount_xarr.loc[self.candidate_symbols, acts]
                )
                # symbol wealth, shape: (n_symbol, )
                # the wealth considered the buy and sell trans fee.
                self.decision_xarr.loc[today, expert_name,
                                       self.candidate_symbols, 'wealth'] = (
                    amount_xarr.loc[self.candidate_symbols, 'wealth']
                )
                self.decision_xarr.loc[today, expert_name,
                                       self.risk_free_symbol, 'wealth'] = (
                    pg_results['risk_free_wealth']
                )

                # portfolio wealth of the expert (considered the trans fee)
                self.portfolio_xarr.loc[today, expert_name, 'wealth'] = (
                        amount_xarr.loc[
                            self.candidate_symbols, 'wealth'].sum()
                        + pg_results['risk_free_wealth']
                )

                # transaction loss
                buy_sum = amount_xarr.loc[
                    self.candidate_symbols, 'buy'].sum()
                sell_sum = amount_xarr.loc[
                    self.candidate_symbols, 'sell'].sum()

                self.portfolio_xarr.loc[today, expert_name, 'tax_loss'] = (
                        buy_sum * self.buy_trans_fee +
                        sell_sum * self.sell_trans_fee
                )

                # risk estimators
                for col in ['CVaR', 'VaR', 'EV_CVaR',
                            'EV_VaR', 'EEV_CVaR', 'VSS']:
                    self.portfolio_xarr.loc[today, expert_name, col] = (
                        pg_results[col]
                    )

            # no-regret strategy, all experts
            self.portfolio_xarr.loc[today, self.expert_names, 'weight'] = (
                self.no_regret_strategy(
                    tdx=tdx,
                    yesterday=self.exp_trans_dates[tdx - 1],
                    today=today,
                    allocated_risk_wealth=allocated_risk_wealth,
                    allocated_risk_free_wealth=allocated_risk_free_wealth
                )
            )
            self.portfolio_xarr.loc[today, "main", 'weight'] = 1

            print("{}, {}-{} expert weights: {}".format(
                today, self.nr_strategy, self.nr_strategy_param,
                self.portfolio_xarr.loc[today, self.expert_names, 'weight']))

            # risky asset of main portfolio,  buy, sell, and wealth
            acts = ['buy', 'sell', 'wealth']
            self.decision_xarr.loc[today, 'main',
                                   self.candidate_symbols, acts] = (
                # shape: (n_expert,), (n_expert, n_symbol, acts)
                # sum=> (n_symbol, acts)
                    self.portfolio_xarr.loc[
                        today, self.expert_names, 'weight'] *
                    self.decision_xarr.loc[today, self.expert_names,
                                           self.candidate_symbols, acts]
            ).sum(axis=0)

            # risk-free asset of the main portfolio
            # shape: (n_expert,), (n_expert, ) => float
            self.decision_xarr.loc[today, 'main',
                                   self.risk_free_symbol, 'wealth'] = (
                    self.portfolio_xarr.loc[
                        today, self.expert_names, 'weight'] *
                    self.decision_xarr.loc[today, self.expert_names,
                                           self.risk_free_symbol, 'wealth']
            ).sum()

            # main portfolio property
            self.portfolio_xarr.loc[today, 'main', 'wealth'] = (
                self.decision_xarr.loc[today, 'main',
                                       self.pf_symbols, 'wealth'].sum()
            )

            # shape: (n_expert,), (n_expert, ) => float
            self.portfolio_xarr.loc[today, 'main', 'tax_loss'] = (
                    self.portfolio_xarr.loc[
                        today, self.expert_names, 'weight'] *
                    self.portfolio_xarr.loc[
                        today, self.expert_names, 'tax_loss']
            ).sum()

            # print(self.portfolio_xarr.loc[today, 'main'])

            # update allocated wealth
            allocated_risk_wealth = self.decision_xarr.loc[
                today, 'main', self.candidate_symbols, 'wealth']
            allocated_risk_free_wealth = self.decision_xarr.loc[
                today, 'main', self.risk_free_symbol, 'wealth']

            # print information
            if tdx % self.print_interval == 0:
                logging.info("{} [{}/{}] {} "
                             "wealth:{:.2f}, {:.3f} secs".format(
                    simulation_name,
                    tdx + 1,
                    self.n_exp_period,
                    today.strftime("%Y%m%d"),
                    float(self.portfolio_xarr.loc[today, 'main', 'wealth']),
                    time() - t1)
                )

        # end of simulation, computing statistics
        edx = self.n_exp_period - 1
        initial_wealth = float(self.initial_risk_wealth.sum() +
                               self.initial_risk_free_wealth)
        final_wealth = float(self.portfolio_xarr.loc[self.exp_end_date, 'main',
                                                     'wealth'])

        # get reports
        reports = self.get_performance_report(
            simulation_name,
            self.group_name,
            self.candidate_symbols,
            self.risk_free_symbol,
            self.exp_start_date,
            self.exp_end_date,
            self.n_exp_period,
            self.buy_trans_fee,
            self.sell_trans_fee,
            float(initial_wealth),
            float(final_wealth),
            self.nr_strategy,
            self.nr_strategy_param,
            self.expert_group_name,
            self.experts,
            self.n_scenario,
            self.decision_xarr,
            self.portfolio_xarr
        )

        # add simulation time
        reports['simulation_time'] = time() - t0

        # write report
        report_path = os.path.join(
            self.report_dir, "report_{}.pkl".format(simulation_name))

        with open(report_path, 'wb') as fout:
            pickle.dump(reports, fout, pickle.HIGHEST_PROTOCOL)

        print("{}-{} {} OK, {:.4f} secs".format(
            platform.node(),
            os.getpid(),
            simulation_name,
            time() - t0)
        )

        return reports


class NIR_SPSP_CVaR(NER_SPSP_CVaR, NIRUtility):
    def __init__(self,
                 str nr_strategy,
                 double nr_strategy_param,
                 str expert_group_name,
                 list experts,
                 str group_name,
                 list candidate_symbols,
                 risk_rois,
                 risk_free_rois,
                 initial_risk_wealth,
                 double initial_risk_free_wealth,
                 double buy_trans_fee=pp.BUY_TRANS_FEE,
                 double sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 int n_scenario=1000,
                 int scenario_set_idx=1,
                 int print_interval=1,
                 report_dir=pp.NRSPSPCVaR_DIR,
                 ):
        """
        no internal regret stage-wise portfolio stochastic programming model
        """
        super(NIR_SPSP_CVaR, self).__init__(
            nr_strategy,
            nr_strategy_param,
            expert_group_name,
            experts,
            group_name,
            candidate_symbols,
            risk_rois,
            risk_free_rois,
            initial_risk_wealth,
            initial_risk_free_wealth,
            buy_trans_fee,
            sell_trans_fee,
            start_date,
            end_date,
            n_scenario,
            scenario_set_idx,
            print_interval,
            report_dir
        )
        # fictitious experts,
        self.virtual_expert_names = [
            "{}-{}".format(s1, s2)
            for s1 in self.expert_names
            for s2 in self.expert_names
            if s1 != s2
        ]
        self.n_virtual_expert = len(self.virtual_expert_names)
        # shape: n_exp_period * n_virtual_expert * n_symbol *  v_properties
        v_properties = ['price_relative', "weight"]
        self.virtual_portfolio_xarr = xr.DataArray(
            np.zeros((self.n_exp_period,
                      self.n_virtual_expert,
                      self.n_expert,
                      len(v_properties)
                      )),
            dims=('trans_date', 'virtual_expert', 'expert', 'decision'),
            coords=(
                self.exp_trans_dates,
                self.virtual_expert_names,
                self.expert_names,
                v_properties
            )
        )


    def no_regret_strategy(self, *args, **kwargs):

        tdx = kwargs['tdx']
        yesterday = kwargs['yesterday']
        today = kwargs['today']
        allocated_risk_wealth = kwargs['allocated_risk_wealth']
        allocated_risk_free_wealth = kwargs['allocated_risk_free_wealth']

        prev_main_wealth = (allocated_risk_wealth.sum() +
                            allocated_risk_free_wealth)
        if tdx == 0:
            # price relatives of all experts (considered trans. fee)
            # shape: (n_expert,)
            self.portfolio_xarr.loc[today, self.expert_names,
                                    'price_relative'] = (
                ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                    today, self.candidate_symbols])
                  ).sum() +
                 allocated_risk_free_wealth * (
                         1 + self.risk_free_rois.loc[today]
                 )) / prev_main_wealth
            )
            # print('{} portfolio_xarr expert relative:{}'.format(today,
            #     self.portfolio_xarr.loc[today, self.expert_names,
            #                             'price_relative']))

            # price relatives of all virtual experts
            # shape: (n_virtual_expert, n_expert)
            self.virtual_portfolio_xarr.loc[today, self.virtual_expert_names,
                self.expert_names, 'price_relative'] = (
                ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                    today, self.candidate_symbols])
                  ).sum() +
                 allocated_risk_free_wealth * (
                         1 + self.risk_free_rois.loc[today]
                 )) / prev_main_wealth
            )
            # print('{} virtual_portfolio_xarr v_expert relative:{}'.format(today,
            #     self.virtual_portfolio_xarr.loc[
            #         today, self.virtual_expert_names, self.expert_names,
            #         'price_relative'] ))

            # price relative of the main portfolio, float
            self.portfolio_xarr.loc[today, 'main', 'price_relative'] = (
                ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                    today, self.candidate_symbols])
                  ).sum() +
                 allocated_risk_free_wealth * (
                         1 + self.risk_free_rois.loc[today]
                 )) / prev_main_wealth
            )
            # print('{} portfolio_xarr main relative:{}'.format(today,
            #      self.portfolio_xarr.loc[today, 'main', 'price_relative']))

            # initial weights of virtual experts,
            # shape: (n_virtual_expert, n_expert)
            self.virtual_portfolio_xarr.loc[today,
                self.virtual_expert_names, self.expert_names, 'weight'] = (
                self.modified_probabilities(
                    np.ones(self.n_expert)/self.n_expert
                )
            )
            # print('{} v_expert weight:{}'.format(today,
            #     self.virtual_portfolio_xarr.loc[today,
            #     self.virtual_expert_names, self.expert_names, 'weight'] ))

            # initial weights of all experts
            return 1. / self.n_expert

        else:
            # tdx >= 1
            # price relatives of all experts, shape: (n_expert, n_symbol)
            prev_risky_wealths = self.decision_xarr.loc[
                yesterday, self.expert_names,
                self.candidate_symbols, 'wealth']

            # shape: (n_expert,)
            prev_riskfree_wealths = self.decision_xarr.loc[
                yesterday, self.expert_names,
                self.risk_free_symbol, 'wealth']

            curr_risky_wealths = (
                prev_risky_wealths *
                (1 + self.exp_risk_rois.loc[today, self.candidate_symbols])
            )

            curr_riskfree_wealths = (
                prev_riskfree_wealths *
                (1 + self.risk_free_rois.loc[today])
            )

            # price relative of all experts, shape: (n_expert,)
            expert_price_relatives = (
                (curr_risky_wealths.sum(axis=1) + curr_riskfree_wealths) /
                (prev_risky_wealths.sum(axis=1) + prev_riskfree_wealths)
            )

            # price relatives of all experts (considered trans. fee)
            self.portfolio_xarr.loc[today, self.expert_names,
                                    'price_relative'] = expert_price_relatives

            # print("curr_risky_wealths:", curr_risky_wealths)
            # print("curr_riskfree_wealths:",  curr_riskfree_wealths)
            # print("expert_price_relatives:", expert_price_relatives)

            # price relative of the main portfolio
            self.portfolio_xarr.loc[today, 'main', 'price_relative'] = (
                    ((allocated_risk_wealth * (1 + self.exp_risk_rois.loc[
                        today, self.candidate_symbols])
                      ).sum() +
                     allocated_risk_free_wealth * (
                             1 + self.risk_free_rois.loc[today]
                     )) / prev_main_wealth
            )

            # price relative of all virtual experts
            # shape: (n_virtual_expert, n_expert)
            self.virtual_portfolio_xarr.loc[today, self.virtual_expert_names,
                self.expert_names, 'price_relative'] = (
                     self.virtual_portfolio_xarr.loc[yesterday,
                        self.virtual_expert_names, self.expert_names,
                        'weight'] *
                    expert_price_relatives
            )

        if self.nr_strategy == 'EXP':
            # cumulative returns of all virtual experts
            # first sum: shape: tdx * n_virtual_expert
            # second sum: shape: n_virtual_expert
            virtual_cum_payoffs = np.log(
                self.virtual_portfolio_xarr.loc[ :today,
                self.virtual_expert_names, self.expert_names,
                'price_relative'].sum(axis=2)
            ).sum(axis=0)

            # exponential predictors
            new_weights = np.exp(self.nr_strategy_param * virtual_cum_payoffs)

            # normalized weights of virtual experts
            virtual_expert_weights = new_weights / new_weights.sum()

        elif self.nr_strategy == 'POLY':
            # shape:  (tdx, n_virtual_expert)
            virtual_time_payoffs = np.log(
                self.virtual_portfolio_xarr.loc[ :today,
                self.virtual_expert_names, self.expert_names,
                'price_relative'].sum(axis=2)
            )
            # shape: (tdx,
            portfoliio_time_payoffs = np.log(self.portfolio_xarr.loc[:today,
                                        'main', 'price_relative'])

            # shape: (n_virtual_expert,)
            diff = (virtual_time_payoffs - portfoliio_time_payoffs).sum(axis=0)

            new_weights = np.power(np.maximum(diff, np.zeros_like(diff)),
                               self.nr_strategy_param - 1)
            virtual_expert_weights = new_weights / new_weights.sum()

        # build column stochastic matrix to get weights of today
        S = self.column_stochastic_matrix(self.n_expert,
                                          virtual_expert_weights.values)
        eigs, eigvs = np.linalg.eig(S)
        normalized_new_weights = (eigvs[:, 0] / eigvs[:, 0].sum()).astype(
            np.float64)

        # record modified strategies of today
        self.virtual_portfolio_xarr.loc[
            today, self.virtual_expert_names, self.expert_names, 'weight'
        ] = self.modified_probabilities(normalized_new_weights)

        # print('{} v_weight:{}'.format(today,
        #    self.virtual_portfolio_xarr.loc[today, self.virtual_expert_names,
        #    self.expert_names, 'weight'] )
        # )

        return normalized_new_weights

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
           simulation name of this experiment
        """

        name = (
            "NIR_SPSP_CVaR_{}_{:.2f}_{}_{}_s{}_sdx{}_{}_{}".format(
                self.nr_strategy,
                self.nr_strategy_param,
                self.group_name,
                self.expert_group_name,
                self.n_scenario,
                self.scenario_set_idx,
                self.exp_start_date.strftime("%Y%m%d"),
                self.exp_end_date.strftime("%Y%m%d"),
            )
        )
        return name

