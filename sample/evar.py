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
    print("primal_CVaR:", instance.cvar_objective())


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
        portfolio_value = sum()
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


def EVaR_original(alpha, rois, solver="ipopt"):
    n_symbol, n_scenario = rois.shape

    # Model
    instance = ConcreteModel("EVaR_original")

    # decision variables
    instance.Z = Var(within=PositiveReals)

    def evar_objective_rule(model):
        mgf = (sum(exp(model.Z * rois[0, sdx])
                   for sdx in range(n_scenario))
               / n_scenario / alpha)
        return 1. / model.Z * log(mgf)

    instance.evar_objective = Objective(rule=evar_objective_rule,
                                        sense=minimize)

    # solve
    opt = SolverFactory(solver)
    # opt.options['max_iter'] = 3000
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # display(instance)
    print("eVaR_original objective:", instance.evar_objective())



def EVaR_right_tail(alpha, rois, solver='ipopt'):
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
    instance.Z = Var(within=PositiveReals)
    instance.Ys = Var(instance.scenarios)

    def weight_constraint_rule(model):
        return sum(model.weights[mdx] for mdx in instance.symbols) == 100.

    instance.weight_constraint = Constraint(rule=weight_constraint_rule)

    def scenario_constraint_rule(model, sdx):
        return model.Ys[sdx] == sum(model.weights[mdx] * model.rois[mdx, sdx]
                                    for mdx in model.symbols)

    instance.scenario_constraint = Constraint(instance.scenarios,
                                              rule=scenario_constraint_rule)

    def evar_objective_rule(model):
        mgf = (sum(exp(model.Z * model.Ys[sdx]) for sdx in
                   model.scenarios)/n_scenario/model.alpha)
        return 1./model.Z * log(mgf)

    instance.evar_objective = Objective(rule=evar_objective_rule,
                                        sense=minimize)

    # solve
    opt = SolverFactory(solver)
    # opt.options['max_iter'] = 3000
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # display(instance)
    print("right eVaR", instance.evar_objective())



def EVaR_left_tail(alpha, rois, solver='ipopt'):
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
    instance.Z = Var(within=PositiveReals)
    instance.Ys = Var(instance.scenarios)

    def weight_constraint_rule(model):
        return sum(model.weights[mdx] for mdx in instance.symbols) == 100.

    instance.weight_constraint = Constraint(rule=weight_constraint_rule)

    def scenario_constraint_rule(model, sdx):
        return model.Ys[sdx] == sum(model.weights[mdx] * model.rois[mdx, sdx]
                                    for mdx in model.symbols)

    instance.scenario_constraint = Constraint(instance.scenarios,
                                              rule=scenario_constraint_rule)

    def evar_objective_rule(model):
        mgf = (1-model.alpha)/(sum(exp(model.Z * model.Ys[sdx]) for sdx in
                   model.scenarios)/n_scenario)
        return 1./model.Z * log(mgf)

    instance.evar_objective = Objective(rule=evar_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    # opt.options['max_iter'] = 3000
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    weights = [instance.weights[mdx].value for mdx in range(n_symbol)]

    # display(instance)
    print("left eVaR", instance.evar_objective())

def HMCR(alpha, rois, p=1, solver='cplex'):
    n_symbol, n_scenario = rois.shape
    # Model
    instance = ConcreteModel()
    instance.rois = rois
    instance.alpha = alpha
    instance.p = p

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

    def hmcr_objective_rule(model):
        if p == 1:
            scenario_exp = (sum(model.Ys[sdx] for sdx in model.scenarios) /
                            n_scenario)
        elif p == 2:
            scenario_exp = (sum(model.Ys[sdx]* model.Ys[sdx]
                                for sdx in model.scenarios) / n_scenario)
            # add 1e-20 to prevent sqrt(0)
            scenario_exp = sqrt(scenario_exp+1e-20)
        elif p == 3:
            # add 1e-20 to prevent sqrt(0)
            scenario_exp = (sum(model.Ys[sdx] * model.Ys[sdx] * model.Ys[sdx]
                                for sdx in model.scenarios) / n_scenario)
            scenario_exp = pow(scenario_exp+1e-20, 1./model.p)

        return model.Z - 1. / (1 - model.alpha) *scenario_exp

    instance.hmcr_objective = Objective(rule=hmcr_objective_rule,
                                        sense=maximize)

    # solve
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.solutions.load_from(results)

    # display(instance)
    print("HMCR P={}:{}".format(p, instance.hmcr_objective()))


def run_VaRs():
    n_symbol, n_scenario = 1, 200
    # rois = np.random.randn(n_symbol, n_scenario)*2
    rois = np.random.lognormal(0, 3, (n_symbol, n_scenario))
    # rois = np.random.gamma(1, 10, (n_symbol, n_scenario))
    # rois = np.random.exponential(0.3, (n_symbol, n_scenario))

    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
        print("alpha = {:.2f}".format(alpha))
        # primal_CVaR(alpha, rois, solver='ipopt')
        primal_CVaR(alpha, rois, solver='cplex')
        # EVaR_original(alpha, rois)
        # EVaR_right_tail(alpha, rois)
        EVaR_left_tail(alpha, rois)
        # HMCR(alpha, rois, p=1, solver='ipopt')
        HMCR(alpha, rois, p=2, solver='ipopt')
        HMCR(alpha, rois, p=3, solver='ipopt')
        # dual_CVaR(alpha, rois)


if __name__ == '__main__':
    run_VaRs()