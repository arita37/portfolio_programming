# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
"""

import portfolio_programming as pp
from portfolio_programming.simulation.wp_base import WeightPortfolio


class PolynomialPortfolio(WeightPortfolio):
    """
    polynomial aggregator strategy
    """

    def __init__(self,
                 poly_power,
                 group_name,
                 symbols,
                 risk_rois,
                 initial_weights,
                 initial_wealth=1e6,
                 buy_trans_fee=pp.BUY_TRANS_FEE,
                 sell_trans_fee=pp.SELL_TRANS_FEE,
                 start_date=pp.EXP_START_DATE,
                 end_date=pp.EXP_END_DATE,
                 print_interval=10):

        super(PolynomialPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval)
        # power degree
        if poly_power < 1:
            raise ValueError('poly power must >= 1, but get {}'.format(poly_power))

        self.poly_power = poly_power


    def get_simulation_name(self, *args, **kwargs):
        return "Pply_{:.2f}_{}_{}_{}".format(
            self.poly_power,
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

    def add_to_reports(self, reports):
        reports['poly_power'] = self.poly_power
        return reports


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

        prev_weights = self.decision_xarr.loc[
            yesterday, self.symbols, 'weight']
        price_relatives = self.exp_rois.loc[today, self.symbols] + 1
        today_prev_weights_sum = (prev_weights * price_relatives).sum()

        new_weights = ()


        new_weights = (prev_weights * np.exp(self.eta * price_relatives /
                                             today_prev_weights_sum))
        normalized_new_weights = new_weights / new_weights.sum()


        return normalized_new_weights
