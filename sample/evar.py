# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import numpy as np
from pyomo.environ import *


def primal_CVaR(alpha, rois, solver='cplex'):
    n_symbol, n_scenario = rois.shape

    # Model
    instance = ConcreteModel()
    instance.rois = rois
    instance.alpha = alpha

    # Set
    instance.symbols = np.arange(n_symbol)
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    instance.weights = Var(instance.symbols, within=NonNegativeReals)
    instance.Z = Var()
    instance.Ys = Var(instance.scenarios, within=NonNegativeReals)

    def weight_constraint_rule(model):
        return sum(model.weights[mdx] for mdx in instance.symbols) == 100.

    instance.weight_constraint = Constraint(rule=weight_constraint_rule)

    def scenario_constraint_rule(model, sdx):
        portfolio_roi = sum(model.weights[mdx] * model.rois[mdx, sdx]
                            for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - portfolio_roi)

    instance.scenario_constraint = Constraint(
        instance.scenarios, rule=scenario_constraint_rule)

    def cvar_objective_rule(model):
        scenario_exp = (sum(model.Ys[sdx] for sdx in model.scenarios) /
                        n_scenario)
        return model.Z - 1. / (1 - model.alpha) * scenario_exp

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # display(instance)
    print("primal_VaR:", instance.Z.value)
    print("primal_CVaR objective:", instance.cvar_objective())


def dual_CVaR(alpha, rois, solver='cplex'):
    n_symbol, n_scenario = rois.shape

    # Model
    instance = ConcreteModel()
    instance.rois = rois
    instance.alpha = alpha

    # Set
    instance.symbols = np.arange(n_symbol)
    instance.scenarios = np.arange(n_scenario)

    # decision variables
    instance.scenario_weights = Var(instance.scenarios, within=NonNegativeReals)
    instance.cvar = Var()

    def scenario_roi_rule(model, mdx):
        scenario_roi = sum(model.rois[mdx, sdx] * model.scenario_weights[sdx]
                           for sdx in model.scenarios)
        return model.cvar  >= scenario_roi

    instance.scenario_roi_constraint = Constraint(
        instance.symbols, rule=scenario_roi_rule)

    def scenario_weight_bound_rule(model, sdx):
        return (model.scenario_weights[sdx] * (1-model.alpha) * n_scenario <=
                100)

    instance.scenario_weight_bound_constraint = Constraint(
        instance.scenarios, rule=scenario_weight_bound_rule)

    def scenario_weight_sum_rule(model):
        return (sum(model.scenario_weights[sdx]
                   for sdx in instance.scenarios) == 100.)

    instance.scenario_weight_sum_constraint = Constraint(
        rule=scenario_weight_sum_rule)

    # objective function
    def cvar_objective_rule(model):
        return model.cvar

    instance.cvar_objective = Objective(rule=cvar_objective_rule,
                                        sense=minimize)
    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # display(instance)
    print("dual_CVaR objective:", instance.cvar_objective())

def EVaR(rois):
    n_symbol, n_scenario = rois.shape


def run_VaRs():
    n_symbol, n_scenario = 1, 500
    rois = np.random.randn(n_symbol, n_scenario)
    for alpha in (0.1, 0.3, 0.6, 0.9):
        print("alpha = {:.2f}".format(alpha))
        primal_CVaR(alpha, rois)
        dual_CVaR(alpha, rois)

if __name__ == '__main__':
    run_VaRs()