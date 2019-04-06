# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>

"""

import numpy as np
import os
import xarray as xr
from time import time
import datetime as dt
import portfolio_programming as pp
# from portfolio_programming.simulation.spsp_cvar import spsp_cvar
from pyomo.environ import *


def spsp_cvar(candidate_symbols: list,
              setting: str,
              max_portfolio_size: int,
              risk_rois,
              risk_free_roi: float,
              allocated_risk_wealth,
              allocated_risk_free_wealth: float,
              buy_trans_fee: float,
              sell_trans_fee: float,
              alpha: float,
              predict_risk_rois,
              predict_risk_free_roi: float,
              n_scenario: int,
              solver="cplex"
              ):
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
    instance.next_portfolio_wealth = Var(instance.scenarios)

    # aux variable, portfolio wealth less than than VaR (Z)
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)

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

    def portfolio_wealth_constraint_rule(model, sdx):
        return (model.next_portfolio_wealth[sdx] ==
                sum((1. + model.predict_risk_rois[mdx, sdx]) *
                                  model.risk_wealth[mdx]
                    for mdx in model.symbols))

    instance.portfolio_wealth_constraint_constraint = Constraint(
        instance.scenarios,
        rule=portfolio_wealth_constraint_rule)

    # common setting constraint
    def scenario_constraint_rule(model, sdx):
        """ auxiliary variable Y depends on scenario. CVaR <= VaR """
        # predict_risk_wealth = sum((1. + model.predict_risk_rois[mdx, sdx]) *
        #                           model.risk_wealth[mdx]
        #                           for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - model.next_portfolio_wealth[sdx])

    instance.scenario_constraint = Constraint(instance.scenarios,
                                              rule=scenario_constraint_rule)

    # additional variables and setting in the general setting
    if setting == "general":
        # aux variable, switching stock variable
        instance.chosen = Var(instance.symbols, within=Binary)

        # general setting constraint
        def chosen_constraint_rule(model, mdx):
            portfolio_wealth = (sum(model.risk_wealth[idx] for idx in
                                    model.symbols) + model.risk_free_wealth)

            # portfolio_wealth = (sum(model.allocated_risk_wealth) +
            #                     model.allocated_risk_free_wealth *
            #                     (1. + model.risk_rois[mdx]))
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

    # logging.DEBUG(display(instance))
    # display(instance)

    # buy and sell amounts
    actions = ['buy', 'sell', 'chosen', 'wealth']
    amounts = xr.DataArray(
        [(instance.buy_amounts[mdx].value,
          instance.sell_amounts[mdx].value,
          -1,
          instance.risk_wealth[mdx].value)
         for mdx in range(n_symbol)],
        dims=('symbol', "action"),
        coords=(candidate_symbols, actions),
    )

    all_scenarios = np.array([instance.Ys[sdx].value
                                 for sdx in range(n_scenario)])

    all_scenarios.sort()
    return {
        "amounts": amounts,
        'scenarios': all_scenarios,
        "risk_free_wealth": instance.risk_free_wealth.value,
        "VaR": instance.Z.value,
        "CVaR": instance.cvar_objective(),
        'alpha': alpha
    }


def reverse_spsp_cvar(candidate_symbols,
                      setting: str,
                      max_portfolio_size: int,
                      risk_rois,
                      risk_free_roi: float,
                      allocated_risk_wealth,
                      allocated_risk_free_wealth: float,
                      buy_trans_fee: float,
                      sell_trans_fee: float,
                      predict_risk_rois,
                      predict_risk_free_roi: float,
                      n_scenario: int,
                      buy_sell_amounts,
                      solver="cplex"
                      ):
    """
    given the buy and sell amounts, to determine the corresponding
    CVaR and alpha
    """


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
    instance.buy_sell_amounts = buy_sell_amounts
    instance.predict_risk_rois = predict_risk_rois
    # shape: (n_stock,)
    instance.mean_predict_risk_rois = predict_risk_rois.mean(axis=1)
    instance.predict_risk_free_roi = predict_risk_free_roi

    # Set
    instance.symbols = np.arange(n_symbol)
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    instance.alpha = Var(within=NonNegativeReals)
    instance.Z = Var()
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)


def test_spsp_cvar(**exp_parameters):

    group_name = exp_parameters['group_name']
    trans_date = exp_parameters['trans_date']
    alpha = exp_parameters['alpha']

    symbols = pp.GROUP_SYMBOLS[group_name]
    n_symbol = len(symbols)
    setting = "compact"
    max_portfolio_size = len(symbols)

    # load data arr
    risky_roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    risk_rois = risky_roi_xarr.loc[
                pp.EXP_START_DATE: pp.EXP_END_DATE, symbols, 'simple_roi']
    risk_rois = risk_rois.loc[trans_date, :].values
    risk_free_roi = 0

    allocated_risk_wealth = xr.DataArray(exp_parameters['allocated_risk_wealth'],
                                         dims=('symbol',), coords=(symbols,))
    allocated_risk_wealth = allocated_risk_wealth.values
    allocated_risk_free_wealth = exp_parameters['allocated_risk_free_wealth']
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425
    # buy_trans_fee = 0
    # sell_trans_fee = 0
    n_scenario = 1000

    # load scenario
    scenario_file = pp.SCENARIO_NAME_FORMAT.format(
        group_name=group_name,
        n_symbol=n_symbol,
        rolling_window_size=100,
        n_scenario=n_scenario,
        sdx=1,
        scenario_start_date=pp.SCENARIO_START_DATE.strftime("%Y%m%d"),
        scenario_end_date=pp.SCENARIO_END_DATE.strftime("%Y%m%d"),
    )
    scenario_path = os.path.join(pp.SCENARIO_SET_DIR, scenario_file)
    scenario_xarr = xr.open_dataarray(scenario_path)
    predict_risk_rois = scenario_xarr.loc[trans_date, symbols, :].values
    # print(predict_risk_rois)
    predict_risk_free_roi = 0

    res = spsp_cvar(symbols, setting, max_portfolio_size,
                    risk_rois, risk_free_roi, allocated_risk_wealth,
                    allocated_risk_free_wealth, buy_trans_fee, sell_trans_fee,
                    alpha, predict_risk_rois, predict_risk_free_roi, n_scenario)
    print("alpha=", alpha)
    print(res['amounts'])
    print(res['amounts'].sum(axis=0))
    print("VaR", res['VaR'])
    print("CVaR", res['CVaR'])
    # print(res['risk_free_wealth'])
    # print(res['scenarios'])
    return res

def valid_spsp_amount(**exp_parameters):
    """
    amounts: xarray.Array
        "amounts": xarray.DataArray, shape:(n_symbol, 4),
            coords: (symbol, ('buy', 'sell','chosen', 'wealth))
    """
    group_name = exp_parameters['group_name']
    trans_date = exp_parameters['trans_date']
    alpha = exp_parameters['alpha']

    res = exp_parameters['res']
    amounts = res['amounts']
    z = res['VaR']
    CVaR = res['CVaR']
    alpha = res['alpha']
    all_ys = res['scenarios']

    symbols = pp.GROUP_SYMBOLS[group_name]
    n_symbol = len(symbols)
    setting = "compact"
    max_portfolio_size = len(symbols)

    # load data arr
    risky_roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    risk_rois = risky_roi_xarr.loc[
                pp.EXP_START_DATE: pp.EXP_END_DATE, symbols, 'simple_roi']
    risk_free_roi = 0

    # load scenario
    n_scenario = 1000
    scenario_file = pp.SCENARIO_NAME_FORMAT.format(
        group_name=group_name,
        n_symbol=n_symbol,
        rolling_window_size=100,
        n_scenario=n_scenario,
        sdx=1,
        scenario_start_date=pp.SCENARIO_START_DATE.strftime("%Y%m%d"),
        scenario_end_date=pp.SCENARIO_END_DATE.strftime("%Y%m%d"),
    )
    scenario_path = os.path.join(pp.SCENARIO_SET_DIR, scenario_file)
    scenario_xarr = xr.open_dataarray(scenario_path)
    predict_risk_rois = scenario_xarr.loc[trans_date, symbols, :]
    # print(predict_risk_rois)
    predict_risk_free_roi = 0

    # valid results
    allocated_risk_wealth = xr.DataArray(exp_parameters['allocated_risk_wealth'],
                                         dims=('symbol',), coords=(symbols,))
    allocated_risk_free_wealth = exp_parameters['allocated_risk_free_wealth']
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425
    # buy_trans_fee = 0
    # sell_trans_fee = 0

    wealths = xr.DataArray(
        np.zeros(n_symbol),
        dims=('symbol'),
        coords=(symbols,),
    )

    for symbol in symbols:
        wealths.loc[symbol] = (
                    (1 + risk_rois.loc[trans_date, symbol]) *
                               allocated_risk_wealth.loc[symbol] +
                               amounts.loc[symbol, 'buy'] -
                               amounts.loc[symbol, 'sell']
                               )

    deposit = (allocated_risk_free_wealth -
               (1+buy_trans_fee) * amounts.loc[symbols, 'buy'].sum() +
               (1-sell_trans_fee) * amounts.loc[symbols, 'sell'].sum())

    print("wealths:", wealths)
    print("deposit:", float(deposit))

    ys =xr.DataArray(
        np.zeros(n_scenario),
    )
    for sdx in range(n_scenario):
        val = (z - (wealths.loc[symbols] *
                       (1 + predict_risk_rois.loc[symbols, sdx])).sum()
                       )
        ys.loc[sdx] = val if val > 0 else 0

    our_cvar = float(z - 1/(1-alpha)/n_scenario*ys.sum())
    print("our CVaR:{:.14}, model CVaR:{:.14}".format(our_cvar, CVaR))


def weighted_spsp(**exp_parameters):
    group_name = exp_parameters['group_name']
    trans_date = exp_parameters['trans_date']

    res = [exp_parameters['res1'], exp_parameters['res2']]
    amounts = [r['amounts'] for r in res]
    zs = [r['VaR'] for r in res]
    CVaRs = [r['CVaR'] for r in res]
    alphas = [r['alpha'] for r in res]
    print(alphas)
    symbols = pp.GROUP_SYMBOLS[group_name]
    n_symbol = len(symbols)
    setting = "compact"
    max_portfolio_size = len(symbols)

    # load data arr
    risky_roi_xarr = xr.open_dataarray(pp.TAIEX_2005_MKT_CAP_NC)
    risk_rois = risky_roi_xarr.loc[
                pp.EXP_START_DATE: pp.EXP_END_DATE, symbols, 'simple_roi']
    risk_free_roi = 0

    # load scenario
    n_scenario = 1000
    scenario_file = pp.SCENARIO_NAME_FORMAT.format(
        group_name=group_name,
        n_symbol=n_symbol,
        rolling_window_size=100,
        n_scenario=n_scenario,
        sdx=1,
        scenario_start_date=pp.SCENARIO_START_DATE.strftime("%Y%m%d"),
        scenario_end_date=pp.SCENARIO_END_DATE.strftime("%Y%m%d"),
    )
    scenario_path = os.path.join(pp.SCENARIO_SET_DIR, scenario_file)
    scenario_xarr = xr.open_dataarray(scenario_path)
    predict_risk_rois = scenario_xarr.loc[trans_date, symbols, :]
    # print(predict_risk_rois)
    predict_risk_free_roi = 0

    # valid results
    allocated_risk_wealth = xr.DataArray(
        exp_parameters['allocated_risk_wealth'],
        dims=('symbol',), coords=(symbols,))
    allocated_risk_free_wealth = exp_parameters['allocated_risk_free_wealth']
    buy_trans_fee = 0.001425
    sell_trans_fee = 0.004425

    wealths = xr.DataArray(
        np.zeros((2, n_symbol)),
        dims=('exp', 'symbol',),
        coords=([0,1], symbols,),
    )
    deposits = np.zeros(2)

    for exp in range(2):
        for symbol in symbols:
            wealths.loc[exp, symbol] = (
                    (1 + risk_rois.loc[trans_date, symbol]) *
                               allocated_risk_wealth.loc[symbol] +
                               amounts[exp].loc[symbol, 'buy'] -
                               amounts[exp].loc[symbol, 'sell']
                               )

        deposits[exp] = (allocated_risk_free_wealth -
               (1 + buy_trans_fee) * amounts[exp].loc[symbols, 'buy'].sum() +
               (1 - sell_trans_fee) * amounts[exp].loc[symbols, 'sell'].sum())

    print(wealths, deposits)

    ys = xr.DataArray(
        np.zeros((2, n_scenario)),
    )
    for exp in range(2):
        for sdx in range(n_scenario):
            val = (zs[exp] - (wealths.loc[exp, symbols] *
                    (1 + predict_risk_rois.loc[symbols, sdx])).sum()
               )
            ys.loc[exp, sdx] = val if val > 0 else 0
        our_cvar = zs[exp] - 1 / (1 - alphas[exp]) / n_scenario * ys.loc[exp, :].sum()
        print("our CVaR:{}, model CVaR:{}".format(
            float(our_cvar), CVaRs[exp]))


def cvar_alpha_plot():
    import matplotlib.pyplot as plt

    xs = np.arange(1, 100)
    alphas = xs / 100.
    cvars = []
    vars = []
    buys = []
    sells = []
    for alpha in alphas:
        res = test_spsp_cvar('TWG1', alpha)
        cvars.append(res['CVaR'])
        vars.append(res['VaR'])
        buys.append(float(res['amounts'].loc['test', 'buy']))
        sells.append(float(res['amounts'].loc['test', 'sell']))

    fig, ax = plt.subplots(4)
    fig.suptitle('Experiment 3', fontsize=24)
    var_ax, cvar_ax, buy_ax, sell_ax = ax[1], ax[0], ax[2], ax[3]

    var_ax.plot(xs, vars)
    var_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    var_ax.set_ylabel('VaR', fontsize=20)
    var_ax.set_xlim(1, 100)
    var_ax.grid(True)

    cvar_ax.plot(xs, cvars)
    cvar_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    cvar_ax.set_ylabel('CVaR', fontsize=20)
    cvar_ax.set_xlim(1, 100)
    cvar_ax.grid(True)

    buy_ax.plot(xs, buys)
    buy_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    buy_ax.set_ylabel('buy amount', fontsize=20)
    buy_ax.set_xlim(1, 100)
    buy_ax.grid(True)

    sell_ax.plot(xs, sells)
    sell_ax.set_xlabel(r'$\alpha$(%)', fontsize=20)
    sell_ax.set_ylabel('sell amount', fontsize=20)
    sell_ax.set_xlim(1, 100)
    sell_ax.grid(True)

    fig.set_size_inches(16, 9)
    fig_path = os.path.join(pp.TMP_DIR, 'cvar_test.png')
    plt.savefig(fig_path, dpi=240, format='png')
    plt.show()


if __name__ == '__main__':
    group_name = 'TWG1'
    trans_date = dt.date(2014, 10, 2)
    alpha = 0.5
    allocated_risk_wealth = np.array([10, 20, 30, 30, 0])
    allocated_risk_free_wealth = 100 - allocated_risk_wealth.sum()

    res = test_spsp_cvar(group_name=group_name, trans_date=trans_date,
                         alpha=alpha,
                         allocated_risk_wealth= allocated_risk_wealth,
                         allocated_risk_free_wealth=allocated_risk_free_wealth)
    #
    # res2 = test_spsp_cvar(group_name=group_name, trans_date=trans_date,
    #                      alpha=alpha + 0.2,
    #                      allocated_risk_wealth= allocated_risk_wealth,
    #
    #                      allocated_risk_free_wealth=allocated_risk_free_wealth)

    # weighted_spsp(group_name=group_name, trans_date=trans_date,
    #                   alpha=alpha, res1=res, res2=res2,
    #                   allocated_risk_wealth=allocated_risk_wealth,
    #                   allocated_risk_free_wealth=allocated_risk_free_wealth)


    # valid_spsp_amount(group_name=group_name, trans_date=trans_date,
    #                   alpha=alpha, res=res,
    #                   allocated_risk_wealth=allocated_risk_wealth,
    #                   allocated_risk_free_wealth=allocated_risk_free_wealth)
    # cvar_alpha_plot()
