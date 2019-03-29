# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import pandas as pd
import numpy as np
import xarray as xr
import scipy.optimize as spopt
import portfolio_programming as pp
from portfolio_programming.simulation.wp_base import WeightPortfolio


class ONSPortfolio(WeightPortfolio):
    """
    online Newton step strategy

     A. Agarwal, E. Hazan, S. Kale, and R. E. Schapire, "Algorithms for
    portfolio management based on the newton method," in Proceedings of
    the 23rd international conference on Machine learning, 2006, pp. 9-16.

    eta = 0, beta = 1, and delta= 0.125.
    """

    def __init__(self,
                 double beta,
                 double delta,
                 str group_name,
                 list symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth = 1e6,
                 double
                 buy_trans_fee = pp.BUY_TRANS_FEE,
                 double
                 sell_trans_fee = pp.SELL_TRANS_FEE,
                 start_date = pp.EXP_START_DATE,
                 end_date = pp.EXP_END_DATE,
                 int
                 print_interval = 10):
        """
        Parameters:
        -------------
        eta : float
            learning rate
        beta: float
            gradient parameters
        delta: float
            heuristic tuning parameter

        """
        super(ONSPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval)


        self.beta = beta
        self.delta = delta

        # save gradient adn Hessian data of the log func.
        self.gradients = xr.DataArray(
             np.zeros((self.n_exp_period, self.n_stock)),
            dims=('trans_date', 'symbol'),
            coords=( self.exp_trans_dates, self.symbols)
        )

        self.Hessians = xr.DataArray(
            np.zeros((self.n_exp_period, self.n_stock, self.n_stock)),
            dims = ('trans_date', 'symbol', 'symbol'),
            coords=( self.exp_trans_dates, self.symbols, self.symbols)
        )


        # get initial gradient and hessians
        init_price_relatives = (self.exp_rois.loc[self.exp_start_date,
                                                    self.symbols] + 1)
        init_grad = init_price_relatives/np.dot(self.initial_weights,
                                                init_price_relatives)
        self.gradients.loc[self.exp_start_date] = init_grad

        init_grad = init_grad.values[:, np.newaxis]
        self.Hessians.loc[self.exp_start_date] = -np.dot(init_grad, init_grad.T)

    def get_simulation_name(self, *args, **kwargs):
         return "ONS_beta{}_delta{}_{}_{}_{}".format(
            self.beta,
            self.delta,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
         )

    def add_to_reports(self, reports):
        reports['beta'] = self.beta
        reports['delta'] = self.delta
        reports['gradient'] = self.gradients
        reports['Hessian'] = self.Hessians
        return reports

    def _weight_projection(self, weights, quad_matrix):
        """
        project weights to simplex domain
        """
        def objective(x):
            return np.dot(np.dot(weights - x, quad_matrix), weights - x)

        # quadratic programing
        projected_weights = spopt.fmin_slsqp(objective, weights,
                           eqcons=[lambda x: x.sum()-1,],
                           bounds = [(0,1) for _ in range(len(weights))],
                           disp = False,
                           )

        return projected_weights

    def get_today_weights(self, *args, **kwargs):
        """
        remaining the same weight as the today_prev_weights

        Parameters: kwargs
        -------------------------
        prev_trans_date=yesterday,
        trans_date=today,
        today_prev_wealth=today_prev_wealth,
        today_prev_portfolio_wealth=today_prev_portfolio_wealth
        """

        yesterday = kwargs['prev_trans_date']
        today = kwargs['trans_date']

        # update gradients and Hessians
        yesterday_weights = self.decision_xarr[yesterday, self.symbols,
                                               'weight']
        price_relatives = (self.exp_rois.loc[today, self.symbols] + 1)
        grad = (price_relatives / np.dot(yesterday_weights, price_relatives))
        self.gradients.loc[today, self.symbols] = grad

        grad = grad.values[:, np.newaxis]
        self.Hessians.loc[today] = -np.dot(grad, grad.T)

        # compute vector b and matrix A
        b_vec = (1. + 1/self.beta) * self.gradients.loc[:today].sum(axis=0)
        a_mtx = (-self.Hessians.loc[:today].sum(axis=0) +
                 np.identity(self.n_symbol))

        # the new weights may out of simplex domain, project it back
        new_weights = self.delta * np.dot(np.linalg.inv(a_mtx), b_vec)
        new_weights = self._weight_projection(new_weights, a_mtx)

        return new_weights