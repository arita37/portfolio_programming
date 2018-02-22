# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False

"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
"""

import os
import platform
from time import time

cimport numpy as cnp
import numpy as np
import pandas as pd
from arch.bootstrap.multiple_comparrison import (SPA, )
from pyomo.environ import *

import portfolio_programming as pp
from portfolio_programming.statistics.risk_adjusted import (
    Sharpe, Sortino_full, Sortino_partial, maximum_drawdown)

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
              str solver=pp.PROG_SOLVER,
              int verbose=False):
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
        "buy_amounts": pandas.Series, shape: (n_stock,)
        "sell_amounts": pandas.Series,  shape: (n_stock,)
        "estimated_var": float
        "estimated_cvar": float
        "estimated_ev_var": float
        "estimated_ev_cvar": float
        "estimated_eev_cvar": float
        "vss": vss, float
        "chosen_symbols": pandas.Series, shape: (n_stock,)

    """
    t0 = time()

    cdef Py_ssize_t n_stock = len(candidate_symbols)

    # Model
    instance = ConcreteModel()
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
    instance.symbols = np.arange(n_stock)
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
        predict_wealth = sum((1. + model.predict_risk_rois[mdx, sdx]) *
                             model.risk_wealth[mdx]
                             for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - predict_wealth)

    instance.scenario_constraint = Constraint(instance.scenarios,
                                              rule=scenario_constraint_rule)

    # additional variables and setting in the general setting
    if setting == "general":
        # aux variable, switching stock variable
        instance.chosen = Var(instance.symbols, within=Binary)

        # general setting constraint
        def chosen_constraint_rule(model, int mdx):
            portfolio_wealth = (sum(model.allocated_risk_wealth) +
                                model.allocated_risk_free_wealth *
                                (1. + model.risk_rois[mdx]))
            return model.risk_wealth[mdx] <= model.chosen[
                mdx] * portfolio_wealth

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

    if verbose:
        display(instance)

    # buy and sell amounts
    buy_amounts = pd.Series([instance.buy_amounts[mdx].value
                             for mdx in range(n_stock)],
                            index=candidate_symbols)
    sell_amounts = pd.Series([instance.sell_amounts[mdx].value
                              for mdx in range(n_stock)],
                             index=candidate_symbols)

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
                    for mdx in np.arange(n_stock)) +
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
        # chosen stock
        chosen_symbols = pd.Series([instance.chosen[mdx].value
                                    for mdx in range(n_stock)],
                                   index=candidate_symbols)
    elif setting == "compact":
        chosen_symbols = pd.Series([1 for mdx in range(n_stock)],
                                   index=candidate_symbols)

    if verbose:
        print("spsp_cvar {} OK, {:.3f} secs".format(setting, time() -t0))

    return {
        "buy_amounts": buy_amounts,
        "sell_amounts": sell_amounts,
        "VaR": estimated_var,
        "CVaR": estimated_cvar,
        "EV_VaR": estimated_ev_var,
        "EV_CVaR": estimated_ev_cvar,
        "EEV_CVaR": estimated_eev_cvar,
        "VSS": vss,
        "chosen_symbols": chosen_symbols,
    }


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


class SPSP_CVaR(ValidMixin):
    def __init__(self,
                 candidate_symbols,
                 str setting,
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
                 int verbose=False):
        """
        stagewise portfolio stochastic programming  model

        Parameters:
        -------------
        candidate_symbols : list of symbols,
            The size of the candidate set is n_stock.

        setting : string,
            {"compact", "general"}

        max_portfolio_size : positive integer
            The max number of stock we can invest in the portfolio.
            The model is the mixed integer linear programming, however,
            if the max_portfolio_size == n_stock, it degenerates to the
            linear programming.

        risk_rois : pandas.DataFrame, shape: (n_period, n_stock)
            The return of all stocks in the given intervals.
            The n_exp_period should be subset of the n_period.

        risk_free_rois : pandas.series or numpy.array, shape: (n_exp_period, )
            The return of risk-free asset, usually all zeros.

        initial_risk_wealth : pandas.series, shape: (n_stock,)
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

        bias_estimator : boolean
            Using biased moment estimators or not.

        report_path : string
            The performance report file path of the simulation.

        verbose : boolean

        Data
        --------------
        wealth_df : pandas.DataFrame, shape: (n_exp_period, n_stock+1)
            The risky and risk-free assets wealth in each period of the
            simulation.
            The risk-free asset symbol is self.risk_free_symbol

        amount_pnl : pandas.Panel, shape: (n_exp_period. n_stock+1, 4)
            Buying, selling, transaction fee and chosen of each asset in each
            period of the simulation.

        estimated_risks: pandas.DataFrame, shape: (n_exp_period, 3)
            The estimated CVaR, VaR and number of gen_scenario_fail in
            the simulation.
            The scenario-generating function may fail in generating
            scenarios in some periods.

        """

        # verify candidate_symbols
        self.valid_dimension("n_stock", len(candidate_symbols),
                             risk_rois.shape[1])

        self.n_stock = len(candidate_symbols)
        self.candidate_symbols = candidate_symbols
        self.periods = risk_rois.index
        self.risk_free_symbol = 'risk_free'

        # verify setting
        if setting not in ("compact", "general"):
            raise (ValueError("Incorrect setting: {}".format(setting)))

        if setting == "compact" and max_portfolio_size != self.n_stock:
            raise (ValueError(
                "The max portfolio size {} must be the same "
                "as the number of symbols {}".format(
                    max_portfolio_size, self.n_stock)))
        self.setting = setting

        # verify max_portfolio_size
        self.valid_nonnegative_value("max_portfolio_size", max_portfolio_size)
        self.max_portfolio_size = max_portfolio_size

        if max_portfolio_size > self.n_stock:
            raise (ValueError(
                "The portfolio size {} cannot large than the "
                "size of candidate set. {}.".format(
                    max_portfolio_size, self.n_stock)))

        # verify risky rois and risk_free_rois
        self.risk_rois = risk_rois
        self.risk_free_rois = risk_free_rois

        # verify initial_wealth
        self.valid_dimension("n_stock", len(candidate_symbols),
                             len(initial_risk_wealth))

        self.valid_nonnegative_list(
            ("initial_risk_wealth", initial_risk_free_wealth))
        self.initial_risk_wealth = initial_risk_wealth

        self.valid_nonnegative_value("initial_risk_free_wealth",
                                     initial_risk_free_wealth)
        self.initial_risk_free_wealth = initial_risk_free_wealth

        # verify transaction fee
        self.valid_range_value("buy_trans_fee", buy_trans_fee, 0, 1)
        self.buy_trans_fee = buy_trans_fee

        self.valid_range_value("sell_trans_fee", sell_trans_fee, 0, 1)
        self.sell_trans_fee = sell_trans_fee

        self.verbose = verbose

        # note .loc() will contain the end_date element
        self.valid_trans_date(start_date, end_date)

        # truncate rois
        self.exp_risk_rois = risk_rois.loc[start_date:end_date]
        self.exp_risk_free_rois = risk_free_rois.loc[
                                  start_date:end_date]

        self.n_exp_period = self.exp_risk_rois.shape[0]
        self.exp_start_date = self.exp_risk_rois.index[0]
        self.exp_end_date = self.exp_risk_rois.index[self.n_exp_period - 1]
        self.exp_start_date_idx = self.risk_rois.index.get_loc(
            self.exp_risk_rois.index[0])

        # verify rolling_window_size
        self.valid_nonnegative_value("rolling_window_size", rolling_window_size)
        if self.exp_start_date_idx < rolling_window_size:
            raise ValueError('There is no enough data.')
        self.rolling_window_size = int(rolling_window_size)

        # verify n_scenario
        self.valid_nonnegative_value("n_scenario", n_scenario)
        self.n_scenario = int(n_scenario)

        # verify alpha
        self.valid_range_value("alpha", alpha, 0, 1)
        self.alpha = float(alpha)

        # load scenario panel, shape:(n_exp_period, n_stock, n_scenario)
        self.scenario_set_idx = scenario_set_idx
        self.scenario_pnl = self.load_generated_scenario()

        # results data
        # wealth DataFrame, shape: (n_exp_period, n_stock+1)
        self.wealth_df = pd.DataFrame(
            np.zeros((self.n_exp_period, self.n_stock)),
            index=self.exp_risk_rois.index,
            columns=candidate_symbols + [self.risk_free_symbol, ]
        )

        # buying,selling, transaction_free, chosen amount panel,
        # shape: (n_exp_period, n_stock+1, 3)
        self.amounts_pnl = pd.Panel(
            np.zeros((self.n_exp_period, self.n_stock + 1, 4)),
            index=self.exp_risk_rois.index,
            major_axis=self.candidate_symbols + [self.risk_free_symbol, ],
            minor_axis=("buy", "sell", "trans_fee", "chosen"),
        )

        # estimated risks, shape(n_exp_period, 6)
        self.estimated_risks = pd.DataFrame(
            np.zeros(self.n_exp_period, 6),
            index=self.exp_risk_rois.index,
            columns=('CVaR', 'VaR', 'EV_CVaR', 'EV_VaR', 'EEV_CVaR', 'VSS')
        )

    def load_generated_scenario(self):
        """
        load generated scenario panel

        Returns
        ---------------
        scenario_pnl: pandas.Panel, shape: (n_exp_period, n_stock, n_scenario)
        """
        name = pp.SCENARIO_NAME_FORMAT.format(
            sdx=self.scenario_set_idx,
            scenario_start_date=pp.SCENARIO_START_DATE.strftime("%Y%m%d"),
            scecnario_end_date=pp.SCENARIO_END_DATE.strftime("%Y%m%d"),
            n_stock=self.n_stock,
            rolling_window_size=self.rolling_window_size,
            n_scenario=self.n_scenario
        )

        scenario_path = os.path.join(pp.SCENARIO_SET_DIR, name)

        if not os.path.exists(scenario_path):
            raise ValueError("{} not exists.".format(scenario_path))
        else:
            scenario_pnl = pd.read_pickle(scenario_path)
            # the experiment interval maybe subset of scenarios.
            if (self.exp_start_date != pp.SCENARIO_START_DATE or
                    self.exp_end_date != pp.SCENARIO_END_DATE):
                scenario_pnl = scenario_pnl.loc[
                               self.exp_start_date:self.exp_end_date]

        return scenario_pnl

    def get_estimated_risk_rois(self, *args, **kwargs):
        """
        estimating next period risky assets rois,

        Returns:
        ----------------------------
        pandas.DataFrame, shape: (n_stock, n_scenario)
        """
        df = self.scenario_pnl.loc[kwargs['trans_date']]
        return df

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
            "buy_amounts": pandas.Series, shape: (n_stock,)
            "sell_amounts": pandas.Series,  shape: (n_stock,)
            "estimated_var": float
            "estimated_cvar": float
            "estimated_ev_var": float
            "estimated_ev_cvar": float
            "estimated_eev_cvar": float
            "vss": vss, float
            "chosen_symbols": pandas.Series, shape: (n_stock,)
        """
        # current exp_period index
        trans_date = kwargs['trans_date']
        results = spsp_cvar(
            self.candidate_symbols,
            self.setting,
            self.max_portfolio_size,
            self.exp_risk_rois.loc[trans_date, :].as_matrix(),
            self.risk_free_rois.loc[trans_date],
            kwargs['allocated_risk_wealth'].as_matrix(),
            kwargs['allocated_risk_free_wealth'],
            self.buy_trans_fee,
            self.sell_trans_fee,
            self.alpha,
            kwargs['estimated_risk_rois'].as_matrix(),
            kwargs['estimated_risk_free_roi'],
            self.n_scenario,
            pp.PROG_SOLVER,
            self.verbose
        )
        return results

    def get_simulation_name(self, *args, **kwargs):
        """
        Returns:
        ------------
        string
           simulation name of this experiment
        """
        name = ("SPSP_CVaR_{}_{}_{}_M{}_Mc{}_h{}_a{:.2f}_s{}_scenario-set-idx{"
                "}".format(
            self.setting,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d"),
            self.max_portfolio_size,
            self.n_stock,
            self.window_length,
            self.alpha,
            self.n_scenario,
            self.scenario_idx)
        )

        return name

    @staticmethod
    def get_performance_report(
            simulation_name,
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
            trans_fee_loss,
            rolling_window_size,
            n_scenario,
            alpha,
            scenario_path,
            wealth_df,
            amount_pnl,
            estimated_risks
    ):
        """
       simulation reports

        Parameters:
        ------------------
        simulation_name : string
        candidate_symbols: list of string
            the candidate symbols in the simulation
        risk_free_symbol: string
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
        amount_pnl : pandas.Panel, shape: (n_exp_period, n_stock+1, 3)
            Buying, selling and transaction fee amount of each asset in each
            period of the simulation.
        estimated_risks: pandas.DataFrame, shape: (n_exp_period, 3)
            The estimated CVaR, VaR and gen_scenario fail count in the
            simulation.
            The scenario-generating function may fail in generating
            scenarios in some periods.
        """
        reports = dict()

        # basic information
        reports['os_uname'] = "|".join(platform.uname())
        reports['simulation_name'] = simulation_name
        reports['candidate_symbols'] = candidate_symbols
        reports['risk_free_symbol'] = risk_free_symbol
        reports['exp_start_date'] = exp_start_date
        reports['exp_end_date'] = exp_end_date
        reports['n_exp_period'] = n_exp_period
        reports['buy_trans_fee'] = buy_trans_fee
        reports['sell_trans_fee'] = sell_trans_fee
        reports['initial_wealth'] = initial_wealth
        reports['final_wealth'] = final_wealth
        reports['trans_fee_loss'] = trans_fee_loss
        reports['wealth_df'] = wealth_df
        reports['amount_pnl'] = amount_pnl
        reports['estimated_risks'] = estimated_risks

        # analysis
        reports['n_stock'] = len(candidate_symbols)
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

        for tdx in range(self.n_exp_period):
            t1 = time.time()
            curr_date = self.exp_risk_rois.index[tdx]

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

            # buy and sell amounts and trans_fee
            self.amounts_pnl.loc[curr_date, :, 'buy'] = pg_results[
                "buy_amounts"]
            self.amounts_pnl.loc[curr_date, :, 'sell'] = pg_results[
                "sell_amounts"]

            # # record the transaction loss
            buy_amounts_sum = pg_results["buy_amounts"].sum()
            sell_amounts_sum = pg_results["sell_amounts"].sum()
            self.trans_fee_loss += (
                    buy_amounts_sum * self.buy_trans_fee +
                    sell_amounts_sum * self.sell_trans_fee
            )

            # buy and sell amounts consider the transaction cost
            total_buy = (buy_amounts_sum * (1 + self.buy_trans_fee))
            total_sell = (sell_amounts_sum * (1 - self.sell_trans_fee))

            # record chosen symbols
            self.amounts_pnl.loc[curr_date, :, 'chosen'] = pg_results[
                'chosen_symbols']

            # capital allocation
            self.wealth_df.loc[curr_date, self.candidate_symbols] = (
                    (1 + self.exp_risk_rois.loc[curr_date]) *
                    allocated_risk_wealth +
                    self.amounts_pnl.loc[curr_date, :, "buy"] -
                    self.amounts_pnl.loc[curr_date, :, "sell"]
            )
            self.wealth_df.loc[curr_date, self.risk_free_symbol] = (
                    (1 + self.exp_risk_free_rois.loc[curr_date]) *
                    allocated_risk_free_wealth -
                    total_buy + total_sell
            )

            # update wealth
            allocated_risk_wealth = self.risk_wealth_df.loc[
                curr_date, self.candidate_symbols]
            allocated_risk_free_wealth = self.risk_free_wealth.loc[
                curr_date, self.risk_free_symbol]

            # record risks
            for col in ("VaR", "CVaR", "EV_VaR", "EV_CVaR", "EEV_CVaR", "VSS"):
                self.estimated_risks.loc[curr_date, col] = pg_results[col]

            # record chosen symbols


            print("{} [{}/{}] {}"
                         "curr_wealth:{:.2f}, {:.3f} secs".format(
                simulation_name,
                tdx + 1,
                self.n_exp_period,
                self.exp_risk_rois.index[tdx].strftime("%Y%m%d"),
                self.wealth_df.loc[tdx].sum(),
                time() - t1)
            )

        # end of simulation, computing statistics
        edx = self.n_exp_period - 1
        initial_wealth =   (self.initial_risk_wealth.sum() + self.initial_risk_free_wealth)
        final_wealth = self.wealth_df.iloc[edx].sum()

        # get reports
        reports = self.get_performance_report(
            simulation_name,
            self.candidate_symbols,
            self.risk_free_symbol,
            self.setting,
            self.max_portfolio_size,
            self.exp_start_date,
            self.exp_end_date,
            self.n_exp_period,
            self.buy_trans_fee,
            self.sell_trans_fee,
            initial_wealth,
            final_wealth,
            self.trans_fee_loss,
            self.rolling_window_size,
            self.n_scenario,
            self.alpha,
            self.rwealth_df,
            self.amounts_pnl,
            self.estimated_risks
        )

        # add simulation time
        reports['simulation_time'] = time() - t0

        print("{} OK [{}-{}], {:.4f} secs".format(
            simulation_name, self.n_stock,
            self.exp_start_date,
            self.exp_end_date,
            time() - t0)
        )

        return reports
