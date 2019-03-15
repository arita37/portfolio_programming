# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>

trans_fee in weight portfolio
"""

import numpy as np
import scipy.optimize as spopt
from time import time

def func_rebalance_opt(
        today_portfolio_wealth,
        prev_weights,
        prev_portfolio_wealth,
        price_relatives,
        today_weights,
        c_buy,
        c_sell
):
    """
    The decision variable must be located in first place.

    Parameters:
    ------------------------
    today_portfolio_wealth: float,
        the portfolio wealth after rebalance, and it is the decision variable

    prev_weights: numpy.array like
        the stocks' weights of yesterday

    prev_portfolio_wealth, : float
        the portfolio wealth of yesterday

    price_relatives : numpy.array like,

    today_weights:   numpy.array like,

    buy: float,
        buy transaction fee

    c_sell: float,
        sell transaction fee

    Returns:
    --------------------
    balance equation
    """
    today_prev_wealths = (
                prev_portfolio_wealth * prev_weights * price_relatives)

    buy_acts = np.maximum(today_portfolio_wealth * today_weights -
                          today_prev_wealths, 0)
    sell_acts = np.maximum(today_prev_wealths -
                           today_portfolio_wealth * today_weights, 0)
    buy_fee = c_buy * buy_acts.sum()
    sell_fee = c_sell * sell_acts.sum()

    balance = (today_portfolio_wealth - today_prev_wealths.sum() +
               buy_fee + sell_fee)
    return balance


def test_func_rebalance_opt(error=1e-6):
    prev_weights = np.array([0.5 ,0.5])
    prev_portfolio_wealth = 100
    price_relatives = np.array([1.02, 0.96])
    today_weights = np.array([0.4, 0.6])
    c_buy = 0.001425
    c_sell = 0.004425
    today_portfolio_wealth = 100
    sol = spopt.newton(func_rebalance_opt, today_portfolio_wealth,
                       args=(prev_weights,
                             prev_portfolio_wealth,
                             price_relatives,
                             today_weights,
                             c_buy,
                             c_sell)
                       )
    today_prev_wealths = (
            prev_portfolio_wealth * prev_weights * price_relatives)
    buy_acts = np.maximum(sol * today_weights - today_prev_wealths, 0)
    sell_acts = np.maximum(today_prev_wealths - sol * today_weights, 0)
    buy_fee = c_buy * buy_acts.sum()
    sell_fee = c_sell * sell_acts.sum()
    balance = (sol - today_prev_wealths.sum() + buy_fee + sell_fee) < error
    return balance


def rand_func_rebalance_opt(n_symbol=5, error=1e-10):
    prev_weights = np.random.dirichlet(np.random.rand(n_symbol))
    prev_portfolio_wealth = 100
    price_relatives = np.random.randn(n_symbol)+1
    today_weights = np.random.dirichlet(np.random.rand(n_symbol))
    c_buy = 0.001425
    c_sell = 0.004425
    today_portfolio_wealth = 100
    sol = spopt.newton(func_rebalance_opt, today_portfolio_wealth,
                       args=(prev_weights,
                             prev_portfolio_wealth,
                             price_relatives,
                             today_weights,
                             c_buy,
                             c_sell)
                       )
    today_prev_wealths = (
            prev_portfolio_wealth * prev_weights * price_relatives)
    buy_acts = np.maximum(sol * today_weights - today_prev_wealths, 0)
    sell_acts = np.maximum(today_prev_wealths - sol * today_weights, 0)
    buy_fee = c_buy * buy_acts.sum()
    sell_fee = c_sell * sell_acts.sum()
    balance = (sol - today_prev_wealths.sum() + buy_fee + sell_fee) < error
    return balance


def test_funfunc_rebalance_opt2(n_symbol=5, error=1e-10):
    t0 = time()
    for _ in range(10000):
        rand_func_rebalance_opt(n_symbol, error)
    print("elapsed: {:.4f} sec".format(time() - t0))


if __name__ == '__main__':
    # test_func_rebalance_opt()
    test_funfunc_rebalance_opt2()