# -*- coding: utf-8 -*-
"""
Author: Hung-Hsin Chen <chen1116@gmail.com>
"""

import portfolio_programming as pp
from portfolio_programming.simulation.wp_base import WeightPortfolio


class BAHPortfolio(WeightPortfolio):
    """
    buy-and-hold portfolio strategy
    """

    def __init__(self,
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
        super(WeightPortfolio, self).__init__(
            group_name, symbols, risk_rois, initial_weights,
            initial_wealth, buy_trans_fee,
            sell_trans_fee, start_date,
            end_date, print_interval)

    def get_simulation_name(self, *args, **kwargs):
        return "BAH_{}_{}_{}".format(
            self.group_name,
            self.exp_start_date.strftime("%Y%m%d"),
            self.exp_end_date.strftime("%Y%m%d")
        )

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
        # normalized weights
        return kwargs['today_prev_wealth'] / kwargs[
            'today_prev_portfolio_wealth']

    def func_rebalance(self, current_portfolio_wealth,
                       prev_weights, prev_portfolio_wealth,
                       price_relatives, today_weights):
        """
        overwrite the balancing function because the BAH does not trading
        in the experimental interval
        """
        return current_portfolio_wealth

