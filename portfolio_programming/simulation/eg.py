# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chen1116@gmail.com>
"""


import os
from datetime import (datetime, date)

import numpy as np
import pandas as pd
from utils import generate_relative_prices_df

from base import WeightTradingPortfolio


class EGPortfolio(WeightTradingPortfolio):
    """
    D. P. Helmbold, R. E. Schapire, Y. Singer, and M. K. Warmuth, "On‐Line
    Portfolio Selection Using Multiplicative Updates," Mathematical Finance,
    vol. 8, pp. 325-347, 1998.
    """

    def __init__(self, symbols, relative_prices, eta, initial_weights=None,
                 initial_wealth=1e6,
                 buy_trans_fee=0.001425, sell_trans_fee=0.004425,
                 start_date=date(2005, 1, 1), end_date=date(2015, 4, 30),
                 verbose=False):
        # setting specific parameters
        self.eta = eta

        # initialize
        super(EGPortfolio, self).__init__(symbols,
                                                         relative_prices,
                                                         initial_weights,
                                                         initial_wealth,
                                                         buy_trans_fee,
                                                         sell_trans_fee,
                                                         start_date, end_date,
                                                         verbose)

    def get_current_weights(self, *args, **kwargs):
        """
        :param eta: float, learning rate of EG algorithm
        :param weights: numpy.array, realized weight of each stock
        :param relative_price: numpy.array, realized relative price of each
        stock
        :return::return:
        """
        tdx = kwargs['tdx']

        weights = self.weights_df.iloc[tdx - 1]
        rp_vec = self.exp_relative_prices.iloc[tdx - 1]

        vec = weights * np.exp(self.eta * rp_vec /
                               np.dot(weights, rp_vec))

        new_weights = vec / vec.sum()

        return new_weights

    def check_specific_parameters(self, *args, **kwargs):
        if self.eta < 0:
            raise ValueError('wrong EG eta parameter: {}'.format(self.eta))

    def get_trading_func_name(self, *args, **kwargs):
        return "{}_{}".format("EG", self.eta)

    def add_results_to_reports(self, reports):
        reports['eta'] = self.eta
        return reports


def run_eg_simulation(n_stock, eta):
    from ipro.dev import (EXP_SYMBOLS, DROPBOX_UP_EXPERIMENT_DIR)

    symbols = EXP_SYMBOLS[:n_stock]
    rp_df = generate_relative_prices_df(symbols)
    start_date = date(2005, 1, 1)
    end_date = date(2015, 4, 30)

    obj = EGPortfolio(symbols, rp_df, eta,
                                     start_date=start_date, end_date=end_date,
                                     verbose=True)
    reports = obj.run()

    # save up report
    exp_rp_df = rp_df.loc[start_date:end_date]
    file_name = '{}_EG_eta_{:.2f}_largest_{}_{}-{}.pkl'.format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        eta,
        len(symbols),
        exp_rp_df.index[0].strftime("%Y%m%d"),
        exp_rp_df.index[-1].strftime("%Y%m%d"))

    file_dir = os.path.join(DROPBOX_UP_EXPERIMENT_DIR, 'eg',
                            'eta_{:.2f}'.format(eta))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    pd.to_pickle(reports, os.path.join(file_dir, file_name))

    return reports


def exponential_gradient(eta, weights, relative_price_vec):
    """
    D. P. Helmbold, R. E. Schapire, Y. Singer, and M. K. Warmuth, "On‐Line
    Portfolio Selection Using Multiplicative Updates," Mathematical Finance,
    vol. 8, pp. 325-347, 1998.

    :param eta: float, learning rate of EG algorithm
    :param weights: numpy.array, realized weight of each stock
    :param relative_price: numpy.array, realized relative price of each stock
    :return:
    """
    vec = weights * np.exp(eta * relative_price_vec /
                           np.dot(weights, relative_price_vec))
    new_weights = vec / vec.sum()
    return new_weights


def internal_regret_exponential_gradient():
    """

    A. Agarwal, E. Hazan, S. Kale, and R. E. Schapire, "Algorithms for
    portfolio management based on the newton method," in Proceedings of
    the 23rd international conference on Machine learning, 2006, pp. 9-16.

    :return:
    """


def modified_exponential_gradient(eta, alpha, weights, relative_price):
    """
   :param eta: float, learning rate of EG algorithm
   :param alpha: float, parameter of EG algorithm
   :param weights: numpy.array,  realized weight of each stock
   :param relative_price: numpy.array, realized relative price of each stock
   :return:
   """


if __name__ == '__main__':
    import sys
    import platform
    import argparse

    sys.path.append(os.path.join(os.path.abspath('..'), '..'))

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n_stock", required=True, type=int,
                        choices=range(5, 55, 5))

    args = parser.parse_args()
    etas = np.arange(5, 105, 5) / 100.
    for eta in etas:
        run_eg_simulation(args.n_stock, eta)
