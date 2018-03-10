# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v3
stagewise mean-variance model
"""

import os
from time import time

import numpy as np
import xarray as xr
from pyomo.environ import *


def mean_variance(symbols, risk_rois, money, risk_factor, solver="cplex"):
    """
    Mean variance to decide the portfolio weight of next stage
    minimize risk_factor * risk  - (1-risk_factor) * mean

    Parameters:
    --------------------------
    symbols: list of string
    risk_rois: numpy.array, shape: (n_symbol, n_historical_period)
    money: float
    risk_factor: float, 0<=risk_factor<=1
        1 means the investor are very conservative
        0 means the investor are very aggressive

    solver: string

    Returns:
    --------------------------
    xarray.DataArray, the portfolio weight according to MV model.

    """
    t0 = time()

    mean_arr = risk_rois.mean(axis=1)
    cov_matrix = np.cov(risk_rois)

    instance = ConcreteModel()

    # Set
    instance.symbols = range(len(symbols))

    # decision variables
    instance.W = Var(instance.symbols, within=NonNegativeReals)

    # constraint
    def money_constraint_rule(model):
        allocation = sum(model.W[idx] for idx in model.symbols)
        return allocation == money

    instance.money_constraint = Constraint(rule=money_constraint_rule)

    # objective
    def min_risk_objective_rule(model):
        profit = sum(model.W[idx] * mean_arr[idx] for idx in model.symbols)
        risk = 0
        for idx in model.symbols:
            for jdx in model.symbols:
                risk += model.W[idx] * model.W[jdx] * cov_matrix[idx, jdx]

        return 0.5 * risk_factor * risk - (1. - risk_factor) * profit

    instance.min_risk_objective = Objective(rule=min_risk_objective_rule,
                                            sense=minimize)

    # Create a solver
    opt = SolverFactory(solver)
    results = opt.solve(instance)
    instance.load(results)
    risk_objective = instance.min_risk_objective()
    weights_xarr = xr.DataArray(
        [instance.W[mdx].values for mdx in range(len(symbols))],
        dims=("symbol",),
        coords=(symbols,)
    )
    display(instance)

    print(risk_objective)
    print(weights_xarr)

    return {
        "objective": risk_objective,
        "weights": weights_xarr,
    }


def testMeanVariance():
    FileDir = os.path.abspath(os.path.curdir)
    PklBasicFeaturesDir = os.path.join(FileDir, '..', 'pkl', 'BasicFeatures')

    symbols = ['2330', '2317', '6505']
    n_period = 100
    ROIs = np.empty((len(symbols), n_period))
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(
            os.path.join(PklBasicFeaturesDir, '%s.pkl' % symbol))
        roi = df['adjROI'][:n_period]
        ROIs[idx] = roi

    mean_variance(symbols, ROIs, money=1e6, risk_weight=1, solver="cplex")


def main():
    recs = iter(sys.stdin.readlines())


if __name__ == '__main__':
    main()
