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
        return sum(model.weights[mdx] for mdx in instance.symbols) == 1.

    instance.weight_constraint = Constraint(rule=weight_constraint_rule)

    def scenario_constraint_rule(model, sdx):
        portfolio_roi = sum(model.weights[mdx] * model.rois[mdx, sdx]
                            for mdx in model.symbols)
        return model.Ys[sdx] >= (model.Z - portfolio_roi)

    instance.scenario_constraint = Constraint(
        instance.scenarios, rule=scenario_constraint_rule)

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
    display(instance)



def dual_CVaR(rois):
    n_symbol, n_scenario = rois.shape



def EVaR(rois):
    n_symbol, n_scenario = rois.shape


def run_VaRs():
    n_symbol, n_scenario = 3, 100
    rois = np.random.randn(n_symbol, n_scenario)
    primal_CVaR(0.5, rois)


if __name__ == '__main__':
    run_VaRs()