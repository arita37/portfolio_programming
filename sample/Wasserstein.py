# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

from pyomo.environ import *


def min_cost_transportation():
    supply = [0.4, 0.6]
    demand = [0.2, 0.5, 0.3]

    instance = ConcreteModel(name="min cost transportation")

    # set
    instance.mines = [0, 1]
    instance.companies = [0, 1, 2]

    # decision variables
    instance.amounts = Var(instance.mines, instance.companies,
                           within=PercentFraction)

    def mine_rule(model, mdx):
        return (supply[mdx] == sum(model.amounts[mdx, cdx]
                                   for cdx in model.companies))

    instance.mine_constraint = Constraint(
        instance.mines, rule=mine_rule)

    def company_rule(model, cdx):
        return (demand[cdx] == sum(model.amounts[mdx, cdx] for mdx in
                                   model.mines))

    instance.company_constraint = Constraint(
        instance.companies, rule=company_rule)

    # objective
    def min_cost_rule(model):
        return sum(abs(supply[mdx] - demand[cdx]) * model.amounts[mdx, cdx]
                   for mdx in model.mines
                   for cdx in model.companies)

    instance.min_cost_objective = Objective(rule=min_cost_rule,
                                            sense=minimize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)


def max_profit_transportation():
    supply = [0.4, 0.6]
    demand = [0.2, 0.5, 0.3]

    instance = ConcreteModel(name="max profit transportation")

    # set
    instance.mines = [0, 1]
    instance.companies = [0, 1, 2]

    # decision variables
    instance.mine_prices = Var(instance.mines,
                               within=PercentFraction)
    instance.company_prices = Var(instance.companies,
                                  within=PercentFraction)

    def price_rule(model, mdx, cdx):
        return ( model.company_prices[cdx] - model.mine_prices[mdx]  <=
                abs(supply[mdx] - demand[cdx]) )

    instance.price_constraint = Constraint(
        instance.mines, instance.companies, rule=price_rule)


    # objective
    def max_profit_rule(model):
        mine_profit = sum(supply[mdx] * model.mine_prices[mdx]
                          for mdx in model.mines)
        company_profit = sum(demand[cdx] * model.company_prices[cdx]
                             for cdx in model.companies)
        return company_profit - mine_profit

    instance.max_profit_objective = Objective(rule=max_profit_rule,
                                              sense=maximize)
    # solve
    opt = SolverFactory("cplex")
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    display(instance)


if __name__ == '__main__':
    # min_cost_transportation()
    max_profit_transportation()
